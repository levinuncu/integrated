from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from ultralytics import YOLO
from ultralytics.engine.results import Results


MODEL_DIR = Path(__file__).resolve().parent / "models"


# Load the model weights from the directory
def load_model(pose: bool = False, *, download: bool = False, fire: bool = False) -> YOLO:
    """Load local COCO, pose or trained Home-Fire weights."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    if pose and fire:
        raise ValueError("The Home-Fire detector has no body keypoints; choose fire or pose.")
    name = "homefire_best.pt" if fire else "yolo26n-pose.pt" if pose else "yolo26n.pt"
    path = MODEL_DIR / name
    if fire and not path.is_file():
        raise FileNotFoundError(f"Missing trained Home-Fire weights: {path}")
    if not path.is_file() and not download:
        raise FileNotFoundError(f"Missing {path}. Run webcam_demo.py --download-models first.")
    return YOLO(str(path))


# Builds a model that can detect persons in a frame
# Arguments enable body keypoints with yolo26n-pose or person tracking with ByteTrack
class PersonDetector:
    """YOLO persons, optionally with Home-Fire smoke/fire or pose keypoints."""

    def __init__(
        self, confidence: float = 0.5, pose: bool = False, track: bool = False, *, fire: bool = False
    ) -> None:
        if not 0 < confidence <= 1:
            raise ValueError("confidence must be greater than 0 and at most 1")
        self.model = load_model(pose, fire=fire)
        names = self.model.names
        required = {"person", "smoke", "fire"} if fire else {"person"}
        if not required.issubset(names.values()):
            raise ValueError(f"Model must contain these classes: {sorted(required)}")
        self.person_class = next(index for index, name in names.items() if name == "person")
        self.classes = [index for index, name in names.items() if name in required]
        self.confidence = confidence
        self.track = track

    # Takes a frame (needs to be adjusted later with real camera frames) and 
    def detect(self, frame: np.ndarray) -> Results:
        """Process a frame and return coordinates in the original image."""
        # classes: people, plus smoke/fire when using the joint trained model.
        # conf: minimum confidence required to keep a detection
        # imgsz: resize to fit a 640x640 target while preserving the aspect ratio
        # verbose=False: disable per-frame inference logs.
        options = dict(classes=self.classes, conf=self.confidence, imgsz=640, verbose=False)
        if self.track:
            # persist: keep tracking state and IDs across consecutive frames.
            # tracker: use the built-in ByteTrack configuration.
            return self.model.track(
                frame, persist=True, tracker="bytetrack.yaml", **options
            )[0]
        return self.model.predict(frame, **options)[0]


# TODO: for later when thermal camera works
class HeatSourceDetector:
    pass


# Creates a dictionary that can be written to the Image Recognition frame
def create_record(
    result: Results, frame_id: int, timestamp_ns: int
) -> dict:
    """Convert tensor results to plain Python values for JSON."""
    # For calculation on cpu if data is on gpu
    result = result.cpu()
    persons = []
    fire_detections, smoke_detections = [], []
    if result.boxes is not None:
        for index, box in enumerate(result.boxes):
            category = result.names[int(box.cls.item())]
            if category in ("fire", "smoke"):
                detections = fire_detections if category == "fire" else smoke_detections
                detections.append({
                    "confidence": round(float(box.conf.item()), 4),
                    "bbox_xyxy": [round(float(x), 2) for x in box.xyxy[0].tolist()],
                })
                continue
            if category != "person":
                continue
            # Add keypoint information when the pose model is enabled.
            # Keypoints follow COCO order: each point is [x, y, confidence]
            keypoints = None
            if result.keypoints is not None:
                keypoints = [
                    [round(float(value), 4) for value in point]
                    for point in result.keypoints.data[index].tolist()
                ]

            # Add person information to image recognition frame
            persons.append(
                {
                    "id": int(box.id.item()) if box.id is not None else None,
                    "confidence": round(float(box.conf.item()), 4), 
                    "bbox_xyxy": [round(float(x), 2) for x in box.xyxy[0].tolist()], # [left, top, right, bottom] in pixels of the original image
                    "keypoints": keypoints,
                    "posture": None,
                    "moving": None, # TODO: Same todo
                    "moving_confidence": None,
                }
            )

    # Add meta information and results for final image recognition frame
    height, width = result.orig_shape
    # TODO: Timestamp functionality needs to be examined on SBC
    timestamp = datetime.fromtimestamp(timestamp_ns / 1_000_000_000, timezone.utc)
    return {
        "schema_version": 2,
        "frame_id": frame_id,
        "timestamp": timestamp.isoformat(timespec="milliseconds"),
        "timestamp_ns": timestamp_ns,
        "time_ms": timestamp_ns // 1_000_000,
        "image_width": width,
        "image_height": height,
        "person_count": len(persons),
        "persons": persons,
        "fire_detected": bool(fire_detections) if "fire" in result.names.values() else None,
        "fire_confidence": max((item["confidence"] for item in fire_detections), default=None),
        "fire_detections": fire_detections,
        "smoke_detected": bool(smoke_detections) if "smoke" in result.names.values() else None,
        "smoke_confidence": max((item["confidence"] for item in smoke_detections), default=None),
        "smoke_detections": smoke_detections,
        "ir_hotspot": None, # TODO: Needs heat detection model
    }
