# Image Recognition Progress Plan

## V1 - Setup, Basic Structure and Demo

### Basic Structure

- `detector.py` loads YOLO26n and converts the results to values that can be saved as JSON
- `webcam_demo.py` reads webcam frames, displays bounding boxes, confidence scores
and FPS, and saves the results in a preliminary JSON schema

### Setup and Usage

```powershell
python -m venv .venv  # Only needed if the virtual environment does not exist yet.
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe webcam_demo.py --download-models
.\.venv\Scripts\python.exe webcam_demo.py
```

The models `yolo26n.pt` and `yolo26n-pose.pt` are stored in `models/` when calling the webcam demo with `--download-models`

Additional options:

- `--camera`: Choose camera; default is 0
- `--confidence`: Change confidence for recognizing a person
- `--pose`: Enables pose tracking at joints
- `--track`: Enables tracking people beyond time frames

### JSON Format, Version 1

Each run creates `runs/detections_<UTC-time>.jsonl`. Frames are different lines in the JSON file. Frames without detections are also saved (`person_count: 0`, `persons: []`).

Current (preliminary) JSON entry example:

```json
{
  "schema_version": 1,
  "frame_id": 0,
  "timestamp": "2026-10-31T12:00:00.000+00:00",
  "timestamp_ns": 1793448000000000000,
  "time_ms": 1793448000000,
  "image_width": 640,
  "image_height": 480,
  "fps": 12.5,
  "person_count": 1,
  "persons": [
    {
      "id": null,
      "confidence": 0.94,
      "bbox_xyxy": [100.0, 50.0, 250.0, 400.0],
      "keypoints": null,
      "posture": null,
      "moving": null
    }
  ],
  "fire_detected": null,
  "fire_confidence": null,
  "ir_hotspot": null
}
```

- `frame_id`: The frame number. Starts at 0 each time the demo starts.
- `timestamp`: The time the program received the frame, before detection starts. Written as a readable date and time in UTC. This is not the exact time the camera took the picture.
- `timestamp_ns`: The same time, as nanoseconds since January 1, 1970 (UTC).
- `time_ms`: The same time, as milliseconds since January 1, 1970 (UTC).
- `bbox_xyxy`: The box around a person: `[left, top, right, bottom]`. All values are pixels in the original image. The top-left corner is `(0, 0)`. x increases to the right, and y increases downward.
- `confidence`: The model's score from 0 to 1. Higher values mean more confidence.
- `id`: The person's tracking ID. Only set when `--track` is enabled.
- `keypoints`: With `--pose`, the 17 body points in a fixed order called COCO order. Each point contains `[x, y, confidence]`. x and y are pixels in the original image. Points with low confidence may have coordinates of 0, so check their scores.
- `posture`, `moving`, `fire_detected`, `fire_confidence`, `ir_hotspot`: Not implemented yet, so the fields contain `null`.

## V2 - Under Construction
[...]

## Next Steps in Image Recognition

- [ ] Define the JSON format for IR frames.
- [ ] Agree with the team on how often to capture, analyze, save and send IR frames to the base station.
- [ ] Add logic to detect whether a person is standing or lying (Pose Estimator).
- [x] Prepare the datasets and keep training, validation and test data separate.
- [ ] Train a model to detect fire in camera images with dataset D-Fire.
