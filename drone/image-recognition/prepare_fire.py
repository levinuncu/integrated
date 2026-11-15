"""Copy Home-Fire train/val, remap fire/smoke to 2/1, add COCO person boxes as 0."""

import argparse
from math import isfinite
import os
from pathlib import Path
import shutil

# Preparation uses local weights and never installs or downloads anything.
os.environ["YOLO_OFFLINE"] = "true"
os.environ["YOLO_AUTOINSTALL"] = "false"

ROOT = Path(__file__).resolve().parent


def fire_labels(path: Path) -> tuple[list[str], int]:
    """Home-Fire: 0=fire, 1=smoke. Clip boxes and drop empty boxes."""
    labels, repairs = [], 0
    for number, row in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not row.strip():
            continue
        values = list(map(float, row.split()))
        if len(values) != 5 or not all(map(isfinite, values)) or values[0] not in (0, 1):
            raise ValueError(f"Invalid Home-Fire label: {path}:{number}")
        category, x, y, width, height = values
        left, right = max(0, x - width / 2), min(1, x + width / 2)
        top, bottom = max(0, y - height / 2), min(1, y + height / 2)
        if right <= left or bottom <= top:
            repairs += 1
            continue
        repairs += int(x - width / 2 < 0 or y - height / 2 < 0 or x + width / 2 > 1 or y + height / 2 > 1)
        target_class = {0: 2, 1: 1}[int(category)]
        labels.append(f"{target_class} {(left + right) / 2:.8f} {(top + bottom) / 2:.8f} "
                      f"{right - left:.8f} {bottom - top:.8f}")
    return labels, repairs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "datasets/Home-Fire")
    parser.add_argument("--output", type=Path, default=ROOT / "datasets/Home-Fire-person")
    parser.add_argument("--confidence", type=float, default=0.15, help="COCO person threshold; default 0.15")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0", help="CUDA device index, or cpu")
    args = parser.parse_args()
    if not isfinite(args.confidence) or not 0 < args.confidence <= 1 or args.imgsz <= 0:
        parser.error("--confidence must be in (0, 1]; --imgsz must be positive")
    source, output = args.source.resolve(), args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}. Use another --output; existing labels are never overwritten.")
    for split in ("train", "val"):
        for folder in ("images", "labels"):
            if not (source / split / folder).is_dir():
                raise FileNotFoundError(source / split / folder)
    weights = ROOT / "models/yolo26n.pt"
    if not weights.is_file():
        raise FileNotFoundError(weights)

    import torch
    import yaml
    from ultralytics import YOLO

    torch.set_num_threads(3)
    model = YOLO(str(weights))
    if model.task != "detect" or len(model.names) != 80 or model.names.get(0) != "person":
        raise ValueError("Expected the original 80-class COCO yolo26n.pt with person=0.")
    output.mkdir(parents=True)
    for split in ("train", "val"):
        images, labels_dir = output / split / "images", output / split / "labels"
        images.mkdir(parents=True)
        labels_dir.mkdir(parents=True)
        count = people = repaired = 0
        results = model.predict(source=str(source / split / "images"), stream=True,
                                classes=[0], conf=args.confidence, imgsz=args.imgsz,
                                device=args.device, verbose=False, save=False)
        for result in results:
            image = Path(result.path)
            labels, fixes = fire_labels(source / split / "labels" / f"{image.stem}.txt")
            boxes = result.boxes.xywhn.cpu().tolist()
            labels.extend("0 " + " ".join(f"{value:.8f}" for value in box) for box in boxes)
            shutil.copy2(image, images / image.name)
            (labels_dir / f"{image.stem}.txt").write_text(
                "".join(row + "\n" for row in labels), encoding="utf-8")
            count, people, repaired = count + 1, people + len(boxes), repaired + fixes
            if count % 100 == 0:
                print(f"{split}: {count} images, {people} person boxes", flush=True)
        if not count:
            raise ValueError(f"No images processed in {split}.")
        print(f"{split}: {count} images, {people} person boxes, {repaired} clipped/removed Home-Fire boxes", flush=True)

    # Only write a training configuration once both splits are complete.
    config = {"path": output.as_posix(), "train": "train/images", "val": "val/images",
              "names": {0: "person", 1: "smoke", 2: "fire"}}
    header = f"# Person pseudo-labels: COCO yolo26n.pt, confidence={args.confidence}, imgsz={args.imgsz}.\n"
    header += "# Home-Fire source: 0=fire, 1=smoke. Target: 0=person, 1=smoke, 2=fire.\n"
    header += "# Automatic labels have NOT been manually reviewed. Original Home-Fire test split is untouched.\n"
    (output / "data.yaml").write_text(header + yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    print(f"Ready: {output / 'data.yaml'}\nCheck the person labels visually; missed people remain unlabelled.")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError) as error:
        raise SystemExit(str(error)) from error
