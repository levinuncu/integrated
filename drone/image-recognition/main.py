"""Run person detection on RGB frames received via ZeroMQ."""

import argparse
import os
import signal
from math import isfinite
from pathlib import Path
from threading import Event
from time import perf_counter, time_ns

import subprocess
import sys

import cv2
import numpy as np
import torch
import zmq

from detector import PersonDetector, create_record, load_model
from frame_wire import TOPIC, FrameMessage, decode_message
from image_posture import ImagePostureClassifier
from record_buffer import RecordBuffer
import prepare_posture


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--address",
        default=os.getenv(
            "CAMERA_FRAMES_ADDRESS",
            "tcp://camera-pipeline:5555",
        ),
        help="ZeroMQ address of the camera pipeline",
    )
    parser.add_argument("--confidence", type=float, default=0.5)
    parser.add_argument("--interval", type=float, default=1.0)
    parser.add_argument("--keep-records", type=int, default=600)
    parser.add_argument("--pose", action="store_true")
    parser.add_argument("--track", action="store_true")
    parser.add_argument("--posture-margin", type=float, default=0.01)
    parser.add_argument("--threads", type=int, default=3)
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Show an OpenCV preview (requires a graphical environment)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "imr_records",
    )
    parser.add_argument("--download-models", action="store_true")

    args = parser.parse_args()

    if not 0 < args.confidence <= 1:
        parser.error("--confidence must be in (0, 1]")
    if not isfinite(args.interval) or args.interval <= 0:
        parser.error("--interval must be finite and positive")
    if args.keep_records <= 0:
        parser.error("--keep-records must be positive")
    if not isfinite(args.posture_margin) or not 0 <= args.posture_margin <= 2:
        parser.error("--posture-margin must be in [0, 2]")
    if args.threads <= 0:
        parser.error("--threads must be positive")

    return args


def create_subscriber(context: zmq.Context, address: str) -> zmq.Socket:
    subscriber = context.socket(zmq.SUB)

    subscriber.setsockopt(zmq.SUBSCRIBE, TOPIC)
    subscriber.setsockopt(zmq.RCVHWM, 2)
    subscriber.setsockopt(zmq.RCVTIMEO, 500)
    subscriber.setsockopt(zmq.LINGER, 0)

    subscriber.connect(address)

    return subscriber


def to_bgr_image(frame: FrameMessage) -> np.ndarray:
    """Convert the RGB888 protocol data to OpenCV's BGR format."""
    rgb = frame.rgb

    expected_size = rgb.width * rgb.height * 3

    if len(rgb.data) != expected_size:
        raise ValueError(
            f"Invalid RGB frame size: expected {expected_size}, "
            f"received {len(rgb.data)}"
        )

    image = np.frombuffer(
        rgb.data,
        dtype=np.uint8,
    ).reshape(rgb.height, rgb.width, 3)

    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


def draw_posture_labels(preview: np.ndarray, record: dict) -> np.ndarray:
    """Draw detected postures and person IDs."""
    height, width = preview.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX

    for person in record["persons"]:
        posture = person.get("posture") or "unknown"

        color = {
            "standing": (100, 230, 100),
            "lying": (0, 165, 255),
            "sitting": (255, 200, 100),
            "crouching": (230, 150, 230),
        }.get(posture, (180, 180, 180))

        left, top, right, bottom = (
            round(value) for value in person["bbox_xyxy"]
        )

        cv2.rectangle(
            preview,
            (left, top),
            (right, bottom),
            color,
            2,
        )

        identity = (
            f"ID {person['id']} | "
            if person.get("id") is not None
            else ""
        )
        text = f"{identity}{posture} | det {person['confidence']:.2f}"

        scale = 0.5
        (text_width, _), _ = cv2.getTextSize(text, font, scale, 1)
        scale *= min(1.0, max(1, width - 8) / max(1, text_width))

        (text_width, text_height), baseline = cv2.getTextSize(
            text, font, scale, 1
        )

        label_height = text_height + baseline + 8
        x = max(0, min(left, width - text_width - 8))
        y = top - label_height if top - label_height >= 44 else max(44, top)
        y = max(0, min(y, height - label_height))

        cv2.rectangle(
            preview,
            (x, y),
            (x + text_width + 8, y + label_height),
            color,
            -1,
        )
        cv2.putText(
            preview,
            text,
            (x + 4, y + text_height + 4),
            font,
            scale,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )

    return preview


def show_preview(result, record: dict, fps: float) -> bool:
    """Display detection results and report whether the user wants to quit."""
    window = "Person detection - Q / Esc to quit"

    preview = result.plot(
        line_width=2,
        boxes=False,
        labels=False,
        conf=False,
    )

    cv2.rectangle(
        preview,
        (0, 0),
        (preview.shape[1], 40),
        (0, 0, 0),
        -1,
    )

    cv2.putText(
        preview,
        f"FPS: {fps:.1f} | Persons: {record['person_count']}",
        (10, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
    )

    cv2.imshow(window, draw_posture_labels(preview, record))

    key = cv2.waitKey(1) & 0xFF

    return (
        key in (ord("q"), ord("Q"), 27)
        or cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1
    )


def main() -> None:
    args = parse_args()

    torch.set_num_threads(args.threads)
    cv2.setNumThreads(1)

if args.download_models:
    load_model(download=True)
    load_model(pose=True, download=True)
    print("YOLO models are ready.", flush=True)

    subprocess.run(
        [
            sys.executable,
            "prepare_posture.py",
            "--viewpoint",
            "drone",
        ],
        check=True,
    )

    print("All models are ready.", flush=True)

    image_posture = ImagePostureClassifier(
        min_margin=args.posture_margin
    )
    detector = PersonDetector(
        args.confidence,
        pose=args.pose,
        track=args.track,
    )
    records = RecordBuffer(
        args.output_dir,
        args.keep_records,
    )

    stop_requested = Event()

    signal.signal(
        signal.SIGTERM,
        lambda *_: stop_requested.set(),
    )
    signal.signal(
        signal.SIGINT,
        lambda *_: stop_requested.set(),
    )

    context = zmq.Context()
    subscriber = create_subscriber(context, args.address)

    print(f"Connected to ZeroMQ publisher: {args.address}", flush=True)
    print(f"Saving detection records to: {records.path}", flush=True)

    next_detection_at = 0.0
    previous_detection_at = None

    try:
        if args.preview:
            cv2.namedWindow(
                "Person detection - Q / Esc to quit",
                cv2.WINDOW_NORMAL,
            )

        while not stop_requested.is_set():
            try:
                parts = subscriber.recv_multipart()
            except zmq.Again:
                continue

            received_at_ns = time_ns()

            try:
                frame = decode_message(parts)
            except (ValueError, KeyError, TypeError) as error:
                print(f"Invalid AI frame: {error}", flush=True)
                continue

            started_at = perf_counter()

            if started_at < next_detection_at:
                continue

            try:
                image = to_bgr_image(frame)
            except ValueError as error:
                print(f"Invalid RGB data: {error}", flush=True)
                continue

            next_detection_at = started_at + args.interval

            # Run the existing person detection model.
            result = detector.detect(image)

            # Generate the detection record.
            record = create_record(
                result,
                frame.rgb.frame_id,
                received_at_ns,
            )

            # Classify each detected person's posture.
            image_posture.apply(image, record)

            # Store the result in the local ring buffer.
            records.append(record)

            completed_at = perf_counter()

            fps = (
                0.0
                if previous_detection_at is None
                else 1.0 / max(
                    completed_at - previous_detection_at,
                    1e-9,
                )
            )
            previous_detection_at = completed_at

            thermal_status = (
                "available" if frame.thermal is not None else "missing"
            )

            print(
                f"Frame #{frame.rgb.frame_id} | "
                f"Persons: {record['person_count']} | "
                f"FPS: {fps:.1f} | "
                f"Thermal: {thermal_status}",
                flush=True,
            )

            if args.preview and show_preview(result, record, fps):
                break

    finally:
        subscriber.close(linger=0)
        context.term()

        if args.preview:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Stopped.")
    except (OSError, RuntimeError, cv2.error, zmq.ZMQError) as error:
        raise SystemExit(str(error)) from error
