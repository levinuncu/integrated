from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from time import time


@dataclass
class RollingRecording:
    directory: Path
    name: str = "rgb"
    segment_seconds: int = 10
    keep_seconds: int = 600
    _session: str = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.segment_seconds <= 0 or self.keep_seconds < self.segment_seconds:
            raise ValueError("Invalid rolling recording settings")

        self._session = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        self.directory.mkdir(parents=True, exist_ok=True)

    @property
    def max_files(self) -> int:
        return (self.keep_seconds + self.segment_seconds - 1) // self.segment_seconds

    @property
    def location_pattern(self) -> str:
        return str(self.directory / f"{self.name}-{self._session}-%05d.mp4")

    def configure(self, sink, nanoseconds_per_second: int) -> None:
        sink.set_property("location", self.location_pattern)
        sink.set_property("max-size-time", self.segment_seconds * nanoseconds_per_second)
        sink.set_property("max-files", self.max_files)
        sink.set_property("async-finalize", True)

    def cleanup(self) -> None:
        files = sorted(
            self.directory.glob(f"{self.name}-*.mp4"),
            key=lambda file: (file.stat().st_mtime, file.name),
        )
        if len(files) <= 1:
            return

        cutoff = time() - self.keep_seconds
        for index, file in enumerate(files[:-1]):
            too_many = len(files) - index > self.max_files
            too_old = file.stat().st_mtime < cutoff
            if too_many or too_old:
                file.unlink(missing_ok=True)
