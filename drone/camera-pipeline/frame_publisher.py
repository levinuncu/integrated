"""Send synchronized frames to the image-recognition service using ZeroMQ.

The GStreamer appsink callback must never wait for an image-recognition model.
For this reason, the callback only enqueues a frame. A dedicated worker owns
and uses the ZeroMQ socket; ZeroMQ sockets must not be shared across threads.
"""

from queue import Empty, Full, Queue
from threading import Event, Thread

import zmq

from frame_bundle import FrameBundle
from frame_wire import (
    FrameMessage,
    RgbFrame,
    ThermalFrame,
    encode_message,
)


class FramePublisher:
    def __init__(self, bind_address: str, queue_size: int = 2) -> None:
        if queue_size < 1:
            raise ValueError("queue_size must be positive")
        self._address = bind_address
        self._queue: Queue[FrameMessage] = Queue(maxsize=queue_size)
        self._stop_event = Event()
        self._ready_event = Event()
        self._thread: Thread | None = None
        self._startup_error: Exception | None = None
        self.dropped_frames = 0
        self.sent_frames = 0

    def start(self) -> None:
        self._thread = Thread(target=self._run_worker, name="frame-publisher", daemon=True)
        self._thread.start()
        if not self._ready_event.wait(timeout=5):
            raise TimeoutError("ZeroMQ publisher did not start")
        if self._startup_error:
            raise RuntimeError("Cannot start ZeroMQ publisher") from self._startup_error
        print(f"AI frame publisher listening at {self._address}", flush=True)

    def publish(self, bundle: FrameBundle) -> None:
        """Enqueue a FrameBundle without waiting for network or AI inference."""
        thermal = bundle.thermal_frame
        message = FrameMessage(
            rgb=RgbFrame(
                camera_id=bundle.rgb_frame.camera_id,
                frame_id=bundle.rgb_frame.frame_id,
                timestamp_ns=bundle.rgb_frame.timestamp_ns,
                width=bundle.rgb_frame.width,
                height=bundle.rgb_frame.height,
                data=bundle.rgb_frame.data,
            ),
            thermal=None if thermal is None else ThermalFrame(
                camera_id=thermal.camera_id,
                frame_id=thermal.frame_id,
                timestamp_ns=thermal.timestamp_ns,
                width=thermal.width,
                height=thermal.height,
                temperatures_c=thermal.data,
            ),
            time_difference_ns=bundle.time_difference_ns,
        )
        try:
            self._queue.put_nowait(message)
        except Full:
            # An old frame is less useful than the newest frame for live AI.
            # Drop it rather than blocking the GStreamer streaming thread.
            try:
                self._queue.get_nowait()
                self.dropped_frames += 1
            except Empty:
                pass
            try:
                self._queue.put_nowait(message)
            except Full:
                self.dropped_frames += 1

    def close(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=3)

    def _run_worker(self) -> None:
        context = zmq.Context()
        socket = context.socket(zmq.PUB)
        try:
            # With only 2 pending messages per subscriber, slow receivers
            # cannot cause unbounded memory usage. PUB drops excess messages.
            socket.setsockopt(zmq.SNDHWM, 2)
            socket.setsockopt(zmq.LINGER, 0)
            socket.bind(self._address)
            self._ready_event.set()

            while not self._stop_event.is_set():
                try:
                    message = self._queue.get(timeout=0.1)
                except Empty:
                    continue
                try:
                    socket.send_multipart(encode_message(message), flags=zmq.NOBLOCK)
                    self.sent_frames += 1
                except zmq.Again:
                    self.dropped_frames += 1
        except Exception as error:
            self._startup_error = error
            self._ready_event.set()
            print(f"AI frame publisher failed: {error}", flush=True)
        finally:
            socket.close(linger=0)
            context.term()
