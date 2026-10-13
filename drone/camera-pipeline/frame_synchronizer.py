"""Time-based pairing of independent RGB and thermal measurements."""

from collections import deque
from threading import Lock

from camera import CameraFrame
from frame_bundle import FrameBundle


class FrameSynchronizer:
    def __init__(
        self,
        max_time_difference_ms: int = 100,
        thermal_buffer_size: int = 32,
    ) -> None:
        if max_time_difference_ms <= 0 or thermal_buffer_size <= 0:
            raise ValueError("Synchronization limits must be positive")

        self._thermal_frames: deque[CameraFrame] = deque(maxlen=thermal_buffer_size)
        self._max_difference_ns = max_time_difference_ms * 1_000_000

        self._lock = Lock()

    def add_thermal_frame(self, frame: CameraFrame) -> None:
        if not isinstance(frame.data, tuple):
            raise TypeError("Expected thermal temperatures as a tuple")
        if len(frame.data) != frame.width * frame.height:
            raise ValueError("IR data size does not match sensor dimensions")

        with self._lock:
            self._thermal_frames.append(frame)

    def pair_with_rgb(self, rgb_frame: CameraFrame) -> FrameBundle:
        if not isinstance(rgb_frame.data, bytes):
            raise TypeError("Expected packed RGB bytes")

        with self._lock:
            closest = min(
                self._thermal_frames,
                key=lambda thermal: abs(thermal.timestamp_ns - rgb_frame.timestamp_ns),
                default=None,
            )

        if closest is None:
            return FrameBundle(rgb_frame, None, None)

        difference_ns = abs(closest.timestamp_ns - rgb_frame.timestamp_ns)
        if difference_ns > self._max_difference_ns:
            return FrameBundle(rgb_frame, None, None)

        return FrameBundle(rgb_frame, closest, difference_ns)
