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


def encode_message(message: FrameMessage) -> list[bytes]:
    _validate_rgb(message.rgb)

    thermal_metadata = None
    thermal_bytes = b""
    if message.thermal is not None:
        _validate_thermal(message.thermal)
        thermal_metadata = {
            "camera_id": message.thermal.camera_id,
            "frame_id": message.thermal.frame_id,
            "timestamp_ns": message.thermal.timestamp_ns,
            "width": message.thermal.width,
            "height": message.thermal.height,
            "format": "float32-le-celsius",
        }
        thermal_bytes = struct.pack(
            f"<{len(message.thermal.temperatures_c)}f",
            *message.thermal.temperatures_c,
        )

    metadata = {
        "version": VERSION,
        "rgb": {
            "camera_id": message.rgb.camera_id,
            "frame_id": message.rgb.frame_id,
            "timestamp_ns": message.rgb.timestamp_ns,
            "width": message.rgb.width,
            "height": message.rgb.height,
            "format": "RGB888",
        },
        "thermal": thermal_metadata,
        "time_difference_ns": message.time_difference_ns,
    }
    return [
        TOPIC,
        json.dumps(metadata, separators=(",", ":")).encode("utf-8"),
        message.rgb.data,
        thermal_bytes,
    ]


def _validate_rgb(frame: RgbFrame) -> None:
    expected_size = frame.width * frame.height * 3
    if expected_size <= 0 or expected_size > MAX_RGB_BYTES:
        raise ValueError("Invalid RGB dimensions")
    if len(frame.data) != expected_size:
        raise ValueError("RGB payload must be packed RGB888")


def _validate_thermal(frame: ThermalFrame) -> None:
    count = frame.width * frame.height
    if count <= 0 or count > MAX_THERMAL_VALUES:
        raise ValueError("Invalid thermal dimensions")
    if len(frame.temperatures_c) != count:
        raise ValueError("Thermal measurements do not match dimensions")
