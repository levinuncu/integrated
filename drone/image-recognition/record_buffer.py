"""Keep the last N ImR records in a persistent JSONL ring buffer."""

import json
import logging
from collections import deque
from pathlib import Path


logger = logging.getLogger(__name__)


class RecordBuffer:
    def __init__(self, directory: Path, keep_records: int) -> None:
        if keep_records <= 0:
            raise ValueError("keep_records must be positive")
        directory.mkdir(parents=True, exist_ok=True)
        self.path = directory / "records.jsonl"
        self._records: deque[dict] = deque(maxlen=keep_records)
        self._save_deferred = False
        if self.path.exists():
            with self.path.open(encoding="utf-8") as source:
                self._records.extend(json.loads(line) for line in source if line.strip())
        self._save()

    def append(self, record: dict) -> None:
        self._records.append(record)
        self._save()

    def _save(self) -> None:
        temporary = self.path.with_suffix(".tmp")
        try:
            with temporary.open("w", encoding="utf-8") as output:
                for record in self._records:
                    output.write(json.dumps(record, allow_nan=False) + "\n")
            temporary.replace(self.path)
        except PermissionError as error:
            # Windows readers can temporarily block writing or replacing files.
            # Retry with the latest buffer on the next append, without stopping AI.
            if getattr(error, "winerror", None) not in (5, 32, 33):
                raise
            if not self._save_deferred:
                logger.warning(
                    "Cannot save ImR records to %s: %s. Keeping up to %d "
                    "measurements in memory and retrying on the next measurement. "
                    "Close programs holding records.jsonl or records.tmp if this persists.",
                    self.path,
                    error,
                    self._records.maxlen,
                )
            self._save_deferred = True
            return
        self._save_deferred = False
