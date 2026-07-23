from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


class SettingsStore:
    def __init__(self, path: str = "stella/data/settings.yaml") -> None:
        self.path = Path(path)

    def load_raw(self) -> dict[str, Any]:
        if not self.path.exists():
            return {}

        with self.path.open("r", encoding="utf-8") as file:
            return yaml.safe_load(file) or {}

    def save_raw(self, data: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)

        with self.path.open("w", encoding="utf-8") as file:
            yaml.safe_dump(
                data,
                file,
                allow_unicode=True,
                sort_keys=False,
            )

    def set_value(self, path: str, value: Any) -> None:
        data = self.load_raw()
        current = data

        parts = path.split(".")

        for part in parts[:-1]:
            if part not in current or not isinstance(current[part], dict):
                current[part] = {}

            current = current[part]

        current[parts[-1]] = value
        self.save_raw(data)