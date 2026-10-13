from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from ultralytics import YOLO
from ultralytics.engine.results import Results


MODEL_DIR = Path(__file__).resolve().parent / "models"


# Load the model weights from the directory
def load_model(pose: bool = False) -> YOLO:
    """Load detection or pose weights from the local models directory."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    name = "yolo26n-pose.pt" if pose else "yolo26n.pt"
    return YOLO(str(MODEL_DIR / name))


# Builds a model that can detect persons in a frame
# Arguments enable body keypoints with yolo26n-pose or person tracking with ByteTrack
class PersonDetector:
    """YOLO26n with optional body keypoints and ByteTrack IDs."""

    def __init__(
        self, confidence: float = 0.5, pose: bool = False, track: bool = False
    ) -> None:
        if not 0 < confidence <= 1:
            raise ValueError("confidence must be greater than 0 and at most 1")
        self.model = load_model(pose)
        self.confidence = confidence
        self.track = track

    # Takes a frame (needs to be adjusted later with real camera frames) and 
    def detect(self, frame: np.ndarray) -> Results:
        """Process a frame and return coordinates in the original image."""
        # classes: return only people (class 0 in both pretrained COCO models)
        # conf: minimum confidence required to keep a detection
        # imgsz: resize to fit a 640x640 target while preserving the aspect ratio
        # verbose=False: disable per-frame inference logs.
        options = dict(classes=[0], conf=self.confidence, imgsz=640, verbose=False)
        if self.track:
            # persist: keep tracking state and IDs across consecutive frames.
            # tracker: use the built-in ByteTrack configuration.
            return self.model.track(
                frame, persist=True, tracker="bytetrack.yaml", **options
            )[0]
        return self.model.predict(frame, **options)[0]


# TODO: Needs self trained model with D-Fire
class FireDetector:
    pass


# TODO: for later when thermal camera works
class HeatSourceDetector:
    pass


# Creates a dictionary that can be written to the Image Recognition frame
def create_record(
    result: Results, frame_id: int, timestamp_ns: int, fps: float
) -> dict:
    """Convert tensor results to plain Python values for JSON."""
    # For calculation on cpu if data is on gpu
    result = result.cpu()
    persons = []
    if result.boxes is not None:
        for index, box in enumerate(result.boxes):
            # Add keypoint information from yolo pose if tracked
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
                    "posture": None, # TODO: Implement estimator for posture from keypoints
                    "moving": None, # TODO: Same todo
                }
            )

    # Add meta information and results for final image recognition frame
    height, width = result.orig_shape
    # TODO: Timestamp functionality needs to be examined on SBC
    timestamp = datetime.fromtimestamp(timestamp_ns / 1_000_000_000, timezone.utc)
    return {
        "schema_version": 1,
        "frame_id": frame_id,
        "timestamp": timestamp.isoformat(timespec="milliseconds"),
        "timestamp_ns": timestamp_ns,
        "time_ms": timestamp_ns // 1_000_000,
        "image_width": width,
        "image_height": height,
        "fps": round(fps, 2), 
        "person_count": len(persons),
        "persons": persons,
        "fire_detected": None, # TODO: Needs fire detection model
        "fire_confidence": None, # TODO: Needs fire detection model
        "ir_hotspot": None, # TODO: Needs heat detection model
    }
