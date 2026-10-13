from dataclasses import dataclass
from camera import CameraFrame


@dataclass(frozen=True)
class FrameBundle:
    rgb_frame: CameraFrame
    thermal_frame: CameraFrame | None
    time_difference_ns: int | None
