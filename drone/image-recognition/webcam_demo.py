import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from time import perf_counter, time_ns

import cv2

from detector import PersonDetector, create_record, load_model

# Arguments for webcam demo
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, default=0, help="Webcam index (default: 0)")
    parser.add_argument("--confidence", type=float, default=0.5, help="Minimum confidence (0, 1]")
    # True or False for pose estimation and ByteTrack tracking
    parser.add_argument("--pose", action="store_true", help="Use YOLO26n-pose with body keypoints")
    parser.add_argument("--track", action="store_true", help="Assign person IDs with ByteTrack")
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path(__file__).resolve().parent / "runs",
        help="Directory for timestamped .jsonl files",
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
    camera = cv2.VideoCapture(args.camera)
    window = "Person detection - Q / Esc to quit"
    try:
        if not camera.isOpened():
            raise RuntimeError(f"Cannot open webcam {args.camera}. Try --camera 1.")

        args.output_dir.mkdir(parents=True, exist_ok=True)
        session = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
        output_path = args.output_dir / f"detections_{session}.jsonl"
        print(f"Saving detections to {output_path}")
        print("Press Q or Esc, close the window, or press Ctrl+C to stop.")

        # Create the file exclusively to protect existing results
        with output_path.open("x", encoding="utf-8") as log:

            # Creates a window for the webcam picture
            cv2.namedWindow(window, cv2.WINDOW_NORMAL)
            frame_id = 0
            previous_frame_time = perf_counter()
            while True:
                # Read a frame from webcam
                ok, frame = camera.read()
                # Timestamp the received frame before running inference
                timestamp_ns = time_ns()
                if not ok:
                    raise RuntimeError("Cannot read a frame from the webcam.")

                # Give camera frame to detector
                result = detector.detect(frame)
                # Ultralytics draws boxes, labels, confidence scores and optional keypoints
                preview = result.plot(line_width=2, labels=True, conf=True)

                # Measure time between processed frames
                now = perf_counter()
                fps = 1.0 / max(now - previous_frame_time, 1e-9) # Calculates frames per second
                previous_frame_time = now

                # Create record
                record = create_record(result, frame_id, timestamp_ns, fps)

                # Each line is a JSON record, including frames without detections
                log.write(json.dumps(record, allow_nan=False) + "\n")
                log.flush() # Pass results on to the os, don't hold them 

                # Shows the webcam picture in the window
                cv2.rectangle(preview, (0, 0), (preview.shape[1], 40), (0, 0, 0), -1)
                cv2.putText(
                    preview, f"FPS: {fps:.1f} | Persons: {record['person_count']}",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2,
                )
                cv2.imshow(window, preview)

                frame_id += 1
                if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q"), 27):
                    break
                if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                    break
    finally:
        # Release the camera and windows after read errors or Ctrl+C as well
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Stopped.")
    except (OSError, RuntimeError, cv2.error) as error:
        raise SystemExit(str(error)) from error
