"""TODO: Remove only for testing."""

from time import monotonic_ns
from camera import Camera, CapturedData


class FakeRgbCamera(Camera):
    def __init__(self, width: int = 640, height: int = 360) -> None:
        super().__init__(camera_id="rgb", width=width, height=height)
        self._color_step = 0

    def _read_measurement(self) -> CapturedData:
        red = (self._color_step * 7) % 256
        self._color_step += 1
        pixel = bytes((red, 128, 255))
        image_bytes = pixel * (self.width * self.height)

        return CapturedData(data=image_bytes, timestamp_ns=monotonic_ns())


class FakeThermalCamera(Camera):
    def __init__(self) -> None:
        super().__init__(camera_id="thermal", width=32, height=24)
        self._hotspot_position = 0

    def _read_measurement(self) -> CapturedData:
        temperatures = [22.5] * (self.width * self.height)
        temperatures[self._hotspot_position] = 38.0
        self._hotspot_position = (self._hotspot_position + 1) % len(temperatures)
        return CapturedData(data=tuple(temperatures), timestamp_ns=monotonic_ns())
