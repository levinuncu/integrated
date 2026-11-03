from time import monotonic_ns

from camera import Camera, CapturedData

# TODO: Instead of returning CapturedData directly use the streams

class RgbCamera(Camera):
    def __init__(self, width: int = 640, height: int = 360) -> None:
        super().__init__(camera_id="rgb", width=width, height=height)

        from picamera2 import Picamera2

        self._camera = Picamera2()

        configuration = self._camera.create_video_configuration(
            main={
                "size": (width, height),
                "format": "BGR888",
            },
        )

        self._camera.configure(configuration)
        self._camera.start()

    def _read_measurement(self) -> CapturedData:
        with self._camera.captured_request() as request:
            image = request.make_array("main")
            timestamp_ns = monotonic_ns()

        return CapturedData(
            data=image.tobytes(),
            timestamp_ns=timestamp_ns,
        )

    def close(self) -> None:
        self._camera.stop()
        self._camera.close()


class ThermalCamera(Camera):
    def __init__(self) -> None:
        super().__init__(camera_id="thermal", width=32, height=24)

        import board
        import busio
        import adafruit_mlx90640

        self._i2c = busio.I2C(
            board.SCL,
            board.SDA,
            frequency=800_000,
        )

        self._camera = adafruit_mlx90640.MLX90640(self._i2c)
        self._camera.refresh_rate = (
            adafruit_mlx90640.RefreshRate.REFRESH_2_HZ
        )

    def _read_measurement(self) -> CapturedData:
        temperatures = [0.0] * (self.width * self.height)

        for _ in range(3):
            try:
                self._camera.getFrame(temperatures)

                return CapturedData(
                    data=tuple(temperatures),
                    timestamp_ns=monotonic_ns(),
                )
            except ValueError:
                continue

        raise RuntimeError("Failed to read MLX90640 frame")

    def close(self) -> None:
        self._i2c.deinit()
