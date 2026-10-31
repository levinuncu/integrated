from dataclasses import dataclass


@dataclass(frozen=True)
class StreamDestination:
    host: str
    port: int = 5004

    def __post_init__(self) -> None:
        if not self.host.strip():
            raise ValueError("RTP destination host must not be empty")
        if not 1 <= self.port <= 65535:
            raise ValueError("RTP port must be between 1 and 65535")
