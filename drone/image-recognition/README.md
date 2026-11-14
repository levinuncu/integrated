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

Each run creates `runs/detections_<UTC-time>.jsonl`. Records are different lines in the JSON file. Records without detections are also saved (`person_count: 0`, `persons: []`).

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

## V2

### Webcam Demo as Mock Main

`webcam_demo.py` is the current baseline. Its `MockCamera` wraps the webcam and returns the same `FrameMessage`/`RgbFrame` format as the camera pipeline, with frame IDs and monotonic capture timestamps. Thermal data is absent. The demo converts the RGB bytes back to a BGR image for the existing detector.

The demo reads webcam frames continuously and analyzes the newest frame at the configured interval. The preview shows the last analyzed frame. `RecordBuffer` keeps only the last N measurements in `drone/data/imr_records/records.jsonl` and restores them on restart. Measurements without detections are saved too.

```powershell
python webcam_demo.py --interval 1 --keep-records 600
```

- `--interval`: Minimum seconds between inference starts; default is 1.
- `--keep-records`: Maximum number of saved measurements; default is 600.
- `--output-dir`: Ring buffer directory; defaults to `drone/data/imr_records` relative to the repository.
- The existing `--camera`, `--confidence`, `--pose`, `--track` and `--download-models` options remain available.

### Image-Based Posture Estimation

`image_posture.py` uses pretrained [MobileCLIP2-S0](https://huggingface.co/timm/MobileCLIP2-S0-OpenCLIP) to compare each YOLO person crop with text descriptions of standing, lying, sitting, crouching and unclear posture. `--pose` is optional; classification uses the image crop. The preview shows the posture, optional tracking ID and person detection confidence (`det`) directly on each person's box. Uncertain postures appear as `unknown`. Image/text similarities remain in the JSON records; they are not shown in a side panel.

`prepare_posture.py` downloads the full model on a PC, encodes three text descriptions per class and exports the image encoder with the averaged text features to `models/posture.pt` (about 45.5 MB). Run the export again after changing the descriptions.

Run from `drone/image-recognition` using the existing `.venv`:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-posture.txt
.\.venv\Scripts\python.exe webcam_demo.py --download-models
.\.venv\Scripts\python.exe prepare_posture.py
.\.venv\Scripts\python.exe webcam_demo.py --track
```

For drone footage, export descriptions covering overhead and oblique views from above:

```powershell
.\.venv\Scripts\python.exe prepare_posture.py --viewpoint drone
```

This overwrites `models/posture.pt` with the selected text prototypes. The default `--viewpoint generic` restores the original descriptions for webcam tests. Prompt changes require re-exporting the model. This changes the text comparisons, without training the image encoder or changing YOLO keypoint detection. Drone prompts have to be tested, once the camera and ai branch is merged.

- `--posture-margin`: Minimum gap between the two highest similarities; default is `0.01`.
- `--threads`: PyTorch CPU threads; default is `3`.
- Posture remains `null` if the unclear-posture class wins, the best similarity is below `0.15`, or the gap is below `--posture-margin`.

Model preparation needs internet. Normal model loading and inference run offline; missing weights cause an error. On the Pi 5, install `requirements.txt` and copy `models/yolo26n.pt` and `models/posture.pt` from the PC. Also copy `models/yolo26n-pose.pt` if using `--pose`. OpenCLIP is only needed for preparation. Person crops are processed one at a time to limit memory use.

The posture classifier still needs validation on drone footage.

### JSON Format, Version 2

Version 2 includes the posture result and classification details. Stored records omit `fps`; the processing interval is configured separately. The webcam demo still displays measured FPS in its preview.

```json
{
  "schema_version": 2,
  "frame_id": 0,
  "timestamp": "2026-10-31T12:00:00.000+00:00",
  "timestamp_ns": 1793448000000000000,
  "time_ms": 1793448000000,
  "image_width": 640,
  "image_height": 480,
  "person_count": 1,
  "persons": [
    {
      "id": null,
      "confidence": 0.94,
      "bbox_xyxy": [100.0, 50.0, 250.0, 400.0],
      "keypoints": null,
      "posture": "standing",
      "moving": null,
      "moving_confidence": null,
      "posture_classification": {
        "candidate": "standing",
        "similarities": {
          "standing": 0.3,
          "lying": 0.1,
          "sitting": 0.22,
          "crouching": 0.19,
          "unknown": 0.17
        },
        "margin": 0.08,
        "crop_xyxy": [85, 15, 265, 435],
        "min_margin": 0.01,
        "min_similarity": 0.15
      }
    }
  ],
  "fire_detected": null,
  "fire_confidence": null,
  "ir_hotspot": null,
  "posture_inference_ms": 86.6
}
```

- `posture`: `standing`, `lying`, `sitting`, `crouching`, or `null` when unclear.
- `posture_classification`: Candidate, similarities, gap between the two best matches, crop coordinates and thresholds. Similarities are not probabilities.
- `posture_inference_ms`: Total crop processing and posture inference time for the measurement, in milliseconds.
- `moving` and `moving_confidence`: Not implemented yet; both contain `null`.
- `frame_id`: RGB frame ID from the source. `MockCamera` increments it for every captured webcam frame, so IDs may have gaps between analyzed frames.
- `timestamp`, `timestamp_ns`, `time_ms`: UTC/Unix reception time of the selected frame, before inference.
- `image_width`, `image_height`, bounding boxes and keypoints refer to the RGB image received by the detector, including any resizing already done by the camera pipeline.

Thermal heat-source detection and movement analysis are still pending. Raw thermal measurements are not stored in ImR records yet.

- [x] Define the JSON format for ImR records.
- [ ] Agree with the team on how often to capture, analyze, save and send ImR records to the base station.
- [x] Add logic to detect whether a person is standing or lying (Pose Estimator).
- [x] Prepare the datasets and keep training, validation and test data separate.
- [ ] Train a model to detect fire in camera images with dataset D-Fire.
