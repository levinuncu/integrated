"""Offline posture classification with the exported MobileCLIP2 image encoder."""

import json
from math import ceil, floor, isfinite
from pathlib import Path
from time import perf_counter

import numpy as np
from PIL import Image
import torch


MODEL_ID = "timm/MobileCLIP2-S0-OpenCLIP"
MODEL_PATH = Path(__file__).resolve().parent / "models" / "posture.pt"


def choose_posture(scores: dict[str, float], min_margin: float, min_similarity: float) -> dict:
    if len(scores) < 2 or not all(isfinite(score) for score in scores.values()):
        return {"posture": None, "candidate": None, "similarities": {}, "margin": None}
    ranked = sorted(scores, key=scores.get, reverse=True)
    candidate = ranked[0]
    margin = scores[candidate] - scores[ranked[1]]
    accepted = candidate != "unknown" and scores[candidate] >= min_similarity and margin >= min_margin
    return {
        "posture": candidate if accepted else None,
        "candidate": candidate,
        "similarities": {label: round(float(score), 4) for label, score in scores.items()},
        "margin": round(float(margin), 4),
    }


def person_crop(frame: np.ndarray, bbox: list[float]) -> tuple[Image.Image | None, list[int] | None]:
    """Keep 10% context and pad to a square so CLIP's center crop preserves the body."""
    if len(bbox) != 4 or not all(isfinite(value) for value in bbox):
        return None, None
    left, top, right, bottom = bbox
    if right <= left or bottom <= top:
        return None, None
    height, width = frame.shape[:2]
    pad_x, pad_y = (right - left) * .1, (bottom - top) * .1
    left, top = max(0, floor(left - pad_x)), max(0, floor(top - pad_y))
    right, bottom = min(width, ceil(right + pad_x)), min(height, ceil(bottom + pad_y))
    if right - left < 32 or bottom - top < 32:
        return None, None
    crop = Image.fromarray(np.ascontiguousarray(frame[top:bottom, left:right, ::-1]))
    size = max(crop.size)
    square = Image.new("RGB", (size, size), (127, 127, 127))
    square.paste(crop, ((size - crop.width) // 2, (size - crop.height) // 2))
    return square, [left, top, right, bottom]


class ImagePostureClassifier:
    def __init__(self, min_margin: float = .01, min_similarity: float = .15,
                 model_path: Path = MODEL_PATH) -> None:
        if not isfinite(min_margin) or not 0 <= min_margin <= 2:
            raise ValueError("Posture margin must be finite and in [0, 2]")
        if not isfinite(min_similarity) or not -1 <= min_similarity <= 1:
            raise ValueError("Minimum similarity must be finite and in [-1, 1]")
        if not model_path.is_file():
            raise FileNotFoundError(f"Missing {model_path}. Run prepare_posture.py on a PC first.")
        metadata = {"metadata.json": b""}
        self.model = torch.jit.load(str(model_path), map_location="cpu", _extra_files=metadata).eval()
        config = json.loads(metadata["metadata.json"])
        self.labels = config["labels"]
        self.image_size = config["image_size"]
        self.min_margin = min_margin
        self.min_similarity = min_similarity

    def apply(self, frame: np.ndarray, record: dict) -> None:
        """Process one person at a time to keep activation memory bounded on the Pi."""
        started = perf_counter()
        for person in record["persons"]:
            person["posture"] = None
            crop, coordinates = person_crop(frame, person["bbox_xyxy"])
            person["posture_classification"] = {
                "candidate": None, "similarities": {}, "margin": None,
                "crop_xyxy": coordinates,
                "min_margin": self.min_margin, "min_similarity": self.min_similarity,
            }
            if crop is not None:
                crop = crop.resize((self.image_size, self.image_size), Image.Resampling.BILINEAR)
                image = torch.from_numpy(np.array(crop)).permute(2, 0, 1).float().div_(255)
                with torch.inference_mode():
                    values = self.model(image.unsqueeze(0))[0].tolist()
                decision = choose_posture(dict(zip(self.labels, values)), self.min_margin, self.min_similarity)
                person["posture"] = decision.pop("posture")
                person["posture_classification"].update(decision)
        record["posture_inference_ms"] = round((perf_counter() - started) * 1000, 1)
