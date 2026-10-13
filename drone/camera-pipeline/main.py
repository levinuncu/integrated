import os

from pathlib import Path

from fake_cameras import FakeRgbCamera, FakeThermalCamera
from pipeline import Pipeline
from frame_publisher import FramePublisher
from stream_destination import StreamDestination
from rolling_recording import RollingRecording


def main() -> None:
    rgb_camera = FakeRgbCamera(width=640, height=360)
    thermal_camera = FakeThermalCamera()

    receiver_host = os.getenv("VIDEO_RECEIVER_HOST", "").strip()
    destination = (
        StreamDestination(
            host=receiver_host,
            port=int(os.getenv("VIDEO_RECEIVER_PORT", "5004")),
        )
        if receiver_host
        else None
    )

    publish_address = os.getenv("AI_PUBLISH_ADDRESS", "tcp://0.0.0.0:5555").strip()
    frame_publisher = FramePublisher(publish_address) if publish_address else None

    pipeline = Pipeline(
        rgb_camera=rgb_camera,
        thermal_camera=thermal_camera,
        rgb_frame_rate=10,
        thermal_frame_rate=8,
        ai_frame_rate=2,
        stream_destination=destination,
        frame_publisher=frame_publisher,
        recording=RollingRecording(
            directory=Path("/recordings"),
            name="rgb",
            segment_seconds=int(os.getenv("RECORD_SEGMENT_SECONDS", "10")),
            keep_seconds=int(os.getenv("RECORD_KEEP_SECONDS", "600")),
        ),
    )

    requested_frames = int(os.getenv("DRONE_FRAME_COUNT", "0"))
    pipeline.run(number_of_frames=requested_frames or None)


if __name__ == "__main__":
    main()
