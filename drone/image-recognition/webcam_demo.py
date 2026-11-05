"""Mock main: webcam frames, person detection and a local ImR ring buffer."""

import argparse
from math import isfinite
from pathlib import Path
from time import monotonic_ns, perf_counter, time_ns

import cv2
import numpy as np

from detector import PersonDetector, create_record, load_model
from frame_wire import FrameMessage, RgbFrame
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
    return args


def main() -> None:
    args = parse_args()

    # Downloads models if not available
    if args.download_models:
        load_model()
        load_model(pose=True)
        print("YOLO26n and YOLO26n-pose are ready in models/.")
        return

    # Load the model before opening the webcam to avoid holding it during downloads
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
                # Ultralytics draws boxes, labels, confidence scores and optional keypoints
                preview = result.plot(line_width=2, labels=True, conf=True)

                # Measure time between processed frames
                now = perf_counter()
                fps = 1.0 / max(now - previous_frame_time, 1e-9) # Calculates frames per second
                previous_frame_time = now

                # Create record
                record = create_record(result, rgb.frame_id, received_at_ns)
                records.append(record)

                # Shows the webcam picture in the window
                cv2.rectangle(preview, (0, 0), (preview.shape[1], 40), (0, 0, 0), -1)
                cv2.putText(
                    preview, f"FPS: {fps:.1f} | Persons: {record['person_count']}",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2,
                )
                cv2.imshow(window, preview)

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
