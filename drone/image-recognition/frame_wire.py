import json
import struct
from dataclasses import dataclass

TOPIC = b"frames"
VERSION = 1
MAX_RGB_BYTES = 24 * 1024 * 1024
MAX_THERMAL_VALUES = 4096


@dataclass(frozen=True)
class RgbFrame:
    camera_id: str
    frame_id: int
    timestamp_ns: int
    width: int
    height: int
    data: bytes


@dataclass(frozen=True)
class ThermalFrame:
    camera_id: str
    frame_id: int
    timestamp_ns: int
    width: int
    height: int
    temperatures_c: tuple[float, ...]


@dataclass(frozen=True)
class FrameMessage:
    rgb: RgbFrame
    thermal: ThermalFrame | None
    time_difference_ns: int | None


def decode_message(parts: list[bytes]) -> FrameMessage:
    if len(parts) != 4 or parts[0] != TOPIC:
        raise ValueError("Expected four ZeroMQ message parts with topic 'frames'")
    if len(parts[1]) > 64 * 1024:
        raise ValueError("Frame metadata exceeds size limit")

    metadata = json.loads(parts[1].decode("utf-8"))
    if metadata.get("version") != VERSION:
        raise ValueError("Unsupported frame protocol version")

    rgb_info = metadata["rgb"]
    if rgb_info.get("format") != "RGB888":
        raise ValueError("Unsupported RGB format")
    rgb = RgbFrame(
        camera_id=str(rgb_info["camera_id"]),
        frame_id=int(rgb_info["frame_id"]),
        timestamp_ns=int(rgb_info["timestamp_ns"]),
        width=int(rgb_info["width"]),
        height=int(rgb_info["height"]),
        data=parts[2],
    )
    _validate_rgb(rgb)

    thermal = None
    thermal_info = metadata.get("thermal")
    if thermal_info is not None:
        if thermal_info.get("format") != "float32-le-celsius":
            raise ValueError("Unsupported thermal format")
        width = int(thermal_info["width"])
        height = int(thermal_info["height"])
        count = width * height
        if count <= 0 or count > MAX_THERMAL_VALUES or len(parts[3]) != count * 4:
            raise ValueError("Invalid thermal payload length")
        temperatures_c = struct.unpack(f"<{count}f", parts[3])
        thermal = ThermalFrame(
            camera_id=str(thermal_info["camera_id"]),
            frame_id=int(thermal_info["frame_id"]),
            timestamp_ns=int(thermal_info["timestamp_ns"]),
            width=width,
            height=height,
            temperatures_c=temperatures_c,
        )
    elif parts[3]:
        raise ValueError("Unexpected thermal bytes without metadata")

    delta = metadata.get("time_difference_ns")
    if delta is not None:
        delta = int(delta)
    if (thermal is None) != (delta is None):
        raise ValueError("Time difference must be present iff thermal data is present")
    if delta is not None and delta != abs(rgb.timestamp_ns - thermal.timestamp_ns):
        raise ValueError("Inconsistent synchronization timestamp difference")

    return FrameMessage(rgb=rgb, thermal=thermal, time_difference_ns=delta)


def _validate_rgb(frame: RgbFrame) -> None:
    expected_size = frame.width * frame.height * 3
    if expected_size <= 0 or expected_size > MAX_RGB_BYTES:
        raise ValueError("Invalid RGB dimensions")
    if len(frame.data) != expected_size:
        raise ValueError("RGB payload must be packed RGB888")
