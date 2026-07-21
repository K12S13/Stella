from __future__ import annotations

from datetime import datetime
from pathlib import Path


class HistoryLogger:
    def __init__(self, path: str = "stella/data/history.log") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, role: str, message: str) -> None:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        with self.path.open("a", encoding="utf-8") as file:
            file.write(f"[{timestamp}] {role}: {message}\n")