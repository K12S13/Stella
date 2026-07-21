from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class Command:
    id: str
    type: str
    phrases: list[str]
    app: str | None = None
    command: str | None = None
    dangerous: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


class CommandRegistry:
    def __init__(self, commands: list[Command]) -> None:
        self.commands = commands

    @classmethod
    def from_yaml(cls, path: str | Path = "stella/data/commands.yaml") -> "CommandRegistry":
        commands_path = Path(path)

        if not commands_path.exists():
            raise FileNotFoundError(f"Commands file not found: {commands_path}")

        with commands_path.open("r", encoding="utf-8") as file:
            data = yaml.safe_load(file) or {}

        raw_commands = data.get("commands", [])
        commands: list[Command] = []

        for item in raw_commands:
            command = Command(
                id=str(item["id"]),
                type=str(item["type"]),
                phrases=[str(phrase).lower().strip() for phrase in item.get("phrases", [])],
                app=item.get("app"),
                command=item.get("command"),
                dangerous=bool(item.get("dangerous", False)),
                metadata={key: value for key, value in item.items() if key not in {
                    "id", "type", "phrases", "app", "command", "dangerous"
                }},
            )
            commands.append(command)

        return cls(commands)

    def all(self) -> list[Command]:
        return self.commands