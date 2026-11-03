"""Hardware-independent camera interface and shared frame data types."""

from abc import ABC, abstractmethod
from dataclasses import dataclass


FrameData = bytes | tuple[float, ...]


@dataclass(frozen=True)
class CapturedData:
    """Raw sensor measurement plus the time the measurement was captured."""

    data: FrameData
    timestamp_ns: int


@dataclass(frozen=True)
class CameraFrame:
    """Public frame format passed to the pipeline and synchronizer."""

    camera_id: str
    frame_id: int
    timestamp_ns: int
    width: int
    height: int
    data: FrameData


class Camera(ABC):
    """Abstract base class for fake and real RGB/IR camera adapters."""

    def __init__(self, camera_id: str, width: int, height: int) -> None:
        if not camera_id:
            raise ValueError("camera_id must not be empty")
        if width <= 0 or height <= 0:
            raise ValueError("Camera dimensions must be positive")

        self._camera_id = camera_id
        self._width = width
        self._height = height
        self._frame_id = 0

    @property
    def width(self) -> int:
        return self._width

    @property
    def height(self) -> int:
        return self._height

    def capture(self) -> CameraFrame:
        measurement = self._read_measurement()
        self._frame_id += 1

        return CameraFrame(
            camera_id=self._camera_id,
            frame_id=self._frame_id,
            timestamp_ns=measurement.timestamp_ns,
            width=self._width,
            height=self._height,
            data=measurement.data,
        )

    @abstractmethod
    def _read_measurement(self) -> CapturedData:
        """Read one image or thermal measurement from the actual source."""

    def close(self) -> None:
        """Release camera resources."""
