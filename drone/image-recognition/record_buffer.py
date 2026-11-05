"""Keep the last N ImR records in a persistent JSONL ring buffer."""

import json
from collections import deque
from pathlib import Path


class RecordBuffer:
    def __init__(self, directory: Path, keep_records: int) -> None:
        if keep_records <= 0:
            raise ValueError("keep_records must be positive")
        directory.mkdir(parents=True, exist_ok=True)
        self.path = directory / "records.jsonl"
        self._records: deque[dict] = deque(maxlen=keep_records)
        if self.path.exists():
            with self.path.open(encoding="utf-8") as source:
                self._records.extend(json.loads(line) for line in source if line.strip())
        self._save()

    def append(self, record: dict) -> None:
        self._records.append(record)
        self._save()

    def _save(self) -> None:
        temporary = self.path.with_suffix(".tmp")
        with temporary.open("w", encoding="utf-8") as output:
            for record in self._records:
                output.write(json.dumps(record, allow_nan=False) + "\n")
        temporary.replace(self.path)
