"""Prepare MobileCLIP2 on a PC; copy the exported weights to the Pi."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from image_posture import MODEL_ID, MODEL_PATH


PROMPTS = {
    "standing": (
        "a photo of a person standing upright",
        "a person standing on their feet",
        "a standing person seen by a camera",
    ),
    "lying": (
        "a photo of a person lying down",
        "a person lying on the ground",
        "a person lying on their back or side",
    ),
    "sitting": (
        "a photo of a person sitting down",
        "a person seated on a chair",
        "a seated person with an upright torso",
    ),
    "crouching": (
        "a photo of a person crouching",
        "a person squatting or kneeling on the ground",
        "a person bending down close to the ground",
    ),
    "unknown": (
        "a close-up of a person's face and shoulders",
        "a person whose body posture is not visible",
        "a partially obscured person with an unclear posture",
    ),
}


# Image encoder model with pre calculated prototypes
class PostureModel(torch.nn.Module):
    def __init__(self, visual: torch.nn.Module, prototypes: torch.Tensor) -> None:
        super().__init__()
        self.visual = visual
        self.register_buffer("prototypes", prototypes)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = torch.nn.functional.normalize(self.visual(images), dim=-1)
        return features @ self.prototypes.T


def main() -> None:
    import open_clip
    from huggingface_hub import hf_hub_download
    from timm.utils import reparameterize_model

    torch.set_num_threads(3)
    # Loads weights and config
    with TemporaryDirectory() as temporary:
        for name in ("open_clip_model.safetensors", "open_clip_config.json"):
            hf_hub_download(MODEL_ID, name, local_dir=temporary,
                            revision="095906d28bf54d7584dc411e8ffe448f34289e05")
        model, _, _ = open_clip.create_model_and_transforms(
            f"local-dir:{Path(temporary).as_posix()}", device="cpu",
        )
        # Turn on eval mode of the model
        model = reparameterize_model(model.eval())
        tokenizer = open_clip.get_tokenizer("MobileCLIP2-S0")
        # Encode prompts
        with torch.inference_mode():
            encoded = model.encode_text(tokenizer([
                text for prompts in PROMPTS.values() for text in prompts
            ]), normalize=True)
            # Prototypes = class vectors
            prototypes = encoded.reshape(len(PROMPTS), 3, -1).mean(dim=1)
            prototypes = torch.nn.functional.normalize(prototypes, dim=-1)
            # Create model
            image_model = PostureModel(model.visual, prototypes).eval()
            # Export finished model
            example = torch.rand(1, 3, 256, 256)
            exported = torch.jit.freeze(torch.jit.trace(image_model, example))
            torch.testing.assert_close(exported(example), image_model(example))

        # Save trained model
        metadata = json.dumps({"model": MODEL_ID, "labels": list(PROMPTS), "image_size": 256})
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        temporary_model = MODEL_PATH.with_suffix(".tmp")
        torch.jit.save(exported, str(temporary_model), _extra_files={"metadata.json": metadata})
        temporary_model.replace(MODEL_PATH)
    print(f"Prepared {MODEL_PATH} ({MODEL_PATH.stat().st_size / 1_000_000:.1f} MB)")


if __name__ == "__main__":
    main()
