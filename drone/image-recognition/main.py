import os
import signal
from threading import Event
from time import sleep

import zmq

from frame_wire import TOPIC, decode_message


def main() -> None:
    address = os.getenv("CAMERA_FRAMES_ADDRESS", "tcp://camera-pipeline:5555")

    stop_requested = Event()
    signal.signal(signal.SIGTERM, lambda *_: stop_requested.set())
    signal.signal(signal.SIGINT, lambda *_: stop_requested.set())

    context = zmq.Context()
    subscriber = context.socket(zmq.SUB)
    subscriber.setsockopt(zmq.SUBSCRIBE, TOPIC)
    subscriber.setsockopt(zmq.RCVHWM, 2)
    subscriber.setsockopt(zmq.RCVTIMEO, 500)
    subscriber.setsockopt(zmq.LINGER, 0)
    subscriber.connect(address)

    previous_frame_id = None
    try:
        while not stop_requested.is_set():
            try:
                parts = subscriber.recv_multipart()
            except zmq.Again:
                continue

            try:
                frame = decode_message(parts)
            except (ValueError, KeyError, TypeError) as error:
                print(f"Invalid AI frame: {error}", flush=True)
                continue

            rgb = frame.rgb
            thermal = frame.thermal
            missed = 0 if previous_frame_id is None else max(0, rgb.frame_id - previous_frame_id - 1)
            previous_frame_id = rgb.frame_id

            thermal_info = (
                "missing"
                if thermal is None
                else f"{thermal.width}x{thermal.height}, hotspot={max(thermal.temperatures_c):.1f}°C"
            )
            print(
                f"Mock AI received RGB #{rgb.frame_id} "
                f"({rgb.width}x{rgb.height}, {len(rgb.data)} bytes), "
                f"thermal={thermal_info}, skipped_since_previous={missed}",
                flush=True,
            )

    finally:
        subscriber.close(linger=0)
        context.term()


if __name__ == "__main__":
    main()
