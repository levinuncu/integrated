"""Mock main: webcam frames, person detection and a local ImR ring buffer."""

import argparse
import os
from math import isfinite
from pathlib import Path
from time import monotonic_ns, perf_counter, time_ns

import cv2
import numpy as np
import torch

# Avoid update checks and automatic package installs during offline operation.
os.environ["YOLO_OFFLINE"] = "true"
os.environ["YOLO_AUTOINSTALL"] = "false"

from detector import PersonDetector, create_record, load_model
from frame_wire import FrameMessage, RgbFrame
from image_posture import ImagePostureClassifier
from record_buffer import RecordBuffer


class MockCamera:
    """Use a webcam to supply the same frame format as the camera pipeline."""

    def __init__(self, camera_index: int) -> None:
        self._camera = cv2.VideoCapture(camera_index)
        self._frame_id = 0
        if not self._camera.isOpened():
            self.close()
            raise RuntimeError(f"Cannot open webcam {camera_index}. Try --camera 1.")

    def capture(self) -> FrameMessage:
        ok, image = self._camera.read()
        timestamp_ns = monotonic_ns()
        if not ok:
            raise RuntimeError("Cannot read a frame from the webcam.")
        height, width = image.shape[:2]
        rgb = RgbFrame(
            camera_id="rgb",
            frame_id=self._frame_id,
            timestamp_ns=timestamp_ns,
            width=width,
            height=height,
            data=image[:, :, ::-1].tobytes(),  # Webcam BGR -> protocol RGB888
        )
        self._frame_id += 1
        return FrameMessage(rgb=rgb, thermal=None, time_difference_ns=None)

    def close(self) -> None:
        self._camera.release()


def draw_posture_labels(preview: np.ndarray, record: dict) -> np.ndarray:
    """Draw each accepted posture at its box; keep uncertain results unknown."""
    height, width = preview.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    for person in record["persons"]:
        posture = person.get("posture") or "unknown"
        color = {
            "standing": (100, 230, 100), "lying": (0, 165, 255),
            "sitting": (255, 200, 100), "crouching": (230, 150, 230),
        }.get(posture, (180, 180, 180))
        left, top, right, bottom = (round(value) for value in person["bbox_xyxy"])
        cv2.rectangle(preview, (left, top), (right, bottom), color, 2)
        identity = f"ID {person['id']} | " if person.get("id") is not None else ""
        text = f"{identity}{posture} | det {person['confidence']:.2f}"
        scale = 0.5
        (text_width, _), _ = cv2.getTextSize(text, font, scale, 1)
        scale *= min(1.0, max(1, width - 8) / max(1, text_width))
        (text_width, text_height), baseline = cv2.getTextSize(text, font, scale, 1)
        label_height = text_height + baseline + 8
        x = max(0, min(left, width - text_width - 8))
        # Prefer above the box; at the top edge place the label inside it.
        y = top - label_height if top - label_height >= 44 else max(44, top)
        y = max(0, min(y, height - label_height))
        cv2.rectangle(preview, (x, y), (x + text_width + 8, y + label_height), color, -1)
        cv2.putText(preview, text, (x + 4, y + text_height + 4),
                    font, scale, (0, 0, 0), 1, cv2.LINE_AA)
    return preview


# Arguments for webcam demo
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, default=0, help="Webcam index (default: 0)")
    parser.add_argument("--confidence", type=float, default=0.5, help="Minimum confidence (0, 1]")
    parser.add_argument("--interval", type=float, default=1.0, help="Seconds between inference starts (default: 1)")
    parser.add_argument("--keep-records", type=int, default=600, help="Maximum number of saved measurements (default: 600)")
    # True or False for pose estimation and ByteTrack tracking
    parser.add_argument("--pose", action="store_true", help="Use YOLO26n-pose with body keypoints")
    parser.add_argument("--track", action="store_true", help="Assign person IDs with ByteTrack")
    parser.add_argument("--posture-margin", type=float, default=0.01,
                        help="Minimum gap between CLIP cosine similarities (default: 0.01)")
    parser.add_argument("--threads", type=int, default=3, help="CPU inference threads (default: 3)")
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "imr_records",
        help="Directory for the records.jsonl ring buffer",
    )
    parser.add_argument(
        "--download-models", action="store_true",
        help="Download both YOLO26n models without opening the camera",
    )
    args = parser.parse_args()
    if args.camera < 0:
        parser.error("--camera must be at least 0")
    if not 0 < args.confidence <= 1:
        parser.error("--confidence must be greater than 0 and at most 1")
    if not isfinite(args.interval) or args.interval <= 0:
        parser.error("--interval must be finite and positive")
    if args.keep_records <= 0:
        parser.error("--keep-records must be positive")
    if not isfinite(args.posture_margin) or not 0 <= args.posture_margin <= 2:
        parser.error("--posture-margin must be finite and in [0, 2]")
    if args.threads <= 0:
        parser.error("--threads must be positive")
    return args


def main() -> None:
    args = parse_args()
    torch.set_num_threads(args.threads)
    cv2.setNumThreads(1)

    # Downloads models if not available
    if args.download_models:
        load_model(download=True)
        load_model(pose=True, download=True)
        print("YOLO26n and YOLO26n-pose are ready in models/.")
        return

    image_posture = ImagePostureClassifier(min_margin=args.posture_margin)
    print("MobileCLIP2-S0 ready on CPU (offline).", flush=True)

    # Check local model files before opening the webcam.
    # Sets up webcam demo
    detector = PersonDetector(args.confidence, pose=args.pose, track=args.track)
    camera = MockCamera(args.camera)
    window = "Person detection - Q / Esc to quit"
    try:
        records = RecordBuffer(args.output_dir, args.keep_records)
        print(f"Keeping {args.keep_records} measurements in {records.path}; interval: {args.interval}s")
        print("Press Q or Esc, close the window, or press Ctrl+C to stop.")

        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        previous_frame_time = perf_counter()
        next_detection_at = 0.0
        while True:
            frame = camera.capture()
            received_at_ns = time_ns()
            started_at = perf_counter()
            if started_at >= next_detection_at:
                next_detection_at = started_at + args.interval
                rgb = frame.rgb
                image = np.frombuffer(rgb.data, dtype=np.uint8).reshape(rgb.height, rgb.width, 3)
                image = np.ascontiguousarray(image[:, :, ::-1])  # Protocol RGB -> detector BGR
                result = detector.detect(image)
                # Keep optional keypoints; draw boxes and posture labels together below.
                preview = result.plot(line_width=2, boxes=False, labels=False, conf=False)

                # Measure time between processed frames
                now = perf_counter()
                fps = 1.0 / max(now - previous_frame_time, 1e-9) # Calculates frames per second
                previous_frame_time = now

                # Create record
                record = create_record(result, rgb.frame_id, received_at_ns)
                image_posture.apply(image, record)
                records.append(record)

                # Shows the webcam picture in the window
                cv2.rectangle(preview, (0, 0), (preview.shape[1], 40), (0, 0, 0), -1)
                cv2.putText(
                    preview, f"FPS: {fps:.1f} | Persons: {record['person_count']}",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2,
                )
                cv2.imshow(window, draw_posture_labels(preview, record))

            if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q"), 27):
                break
            if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        # Release the camera and windows after read errors or Ctrl+C as well
        camera.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Stopped.")
    except (OSError, RuntimeError, cv2.error) as error:
        raise SystemExit(str(error)) from error
