"""Prepare MobileCLIP2 on a PC; copy the exported weights to the Pi."""

import argparse
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


DRONE_PROMPTS = {
    "standing": (
        "a drone photo of a person standing upright, viewed from above",
        "an overhead view of a person standing on their feet",
        "an aerial photo of a standing person, viewed obliquely from above",
    ),
    "lying": (
        "a drone photo of a person lying down, viewed from above",
        "an overhead view of a person lying on the ground",
        "an aerial photo of a person lying on their back or side, viewed obliquely from above",
    ),
    "sitting": (
        "a drone photo of a person sitting down, viewed from above",
        "an overhead view of a person seated on the ground or on a chair",
        "an aerial photo of a seated person, viewed obliquely from above",
    ),
    "crouching": (
        "a drone photo of a person crouching, viewed from above",
        "an overhead view of a person squatting or kneeling on the ground",
        "an aerial photo of a crouching person, viewed obliquely from above",
    ),
    "unknown": (
        "a drone photo of a partially obscured person with an unclear posture, viewed from above",
        "an overhead view of a person whose body posture is not visible",
        "an aerial photo of a person whose posture cannot be determined, viewed obliquely from above",
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--viewpoint", choices=("generic", "drone"), default="generic",
                        help="Text descriptions to export (default: generic for webcam tests)")
    args = parser.parse_args()
    prompts = DRONE_PROMPTS if args.viewpoint == "drone" else PROMPTS

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
                text for descriptions in prompts.values() for text in descriptions
            ]), normalize=True)
            # Prototypes = class vectors
            prototypes = encoded.reshape(len(prompts), 3, -1).mean(dim=1)
            prototypes = torch.nn.functional.normalize(prototypes, dim=-1)
            # Create model
            image_model = PostureModel(model.visual, prototypes).eval()
            # Export finished model
            example = torch.rand(1, 3, 256, 256)
            exported = torch.jit.freeze(torch.jit.trace(image_model, example))
            torch.testing.assert_close(exported(example), image_model(example))

        # Save trained model
        metadata = json.dumps({"model": MODEL_ID, "labels": list(prompts), "image_size": 256,
                               "viewpoint": args.viewpoint, "prompts": prompts})
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        temporary_model = MODEL_PATH.with_suffix(".tmp")
        torch.jit.save(exported, str(temporary_model), _extra_files={"metadata.json": metadata})
        temporary_model.replace(MODEL_PATH)
    print(f"Prepared {MODEL_PATH} ({MODEL_PATH.stat().st_size / 1_000_000:.1f} MB), "
          f"viewpoint={args.viewpoint}")


if __name__ == "__main__":
    main()
