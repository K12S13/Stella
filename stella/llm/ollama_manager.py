from __future__ import annotations

import shutil
import subprocess


class OllamaManager:
    def is_available(self) -> bool:
        return shutil.which("ollama") is not None

    def installed_models(self) -> list[str]:
        if not self.is_available():
            return []

        result = subprocess.run(
            ["ollama", "list"],
            text=True,
            capture_output=True,
            check=False,
        )

        if result.returncode != 0:
            return []

        models: list[str] = []

        for index, line in enumerate(result.stdout.splitlines()):
            if index == 0:
                continue

            parts = line.split()
            if parts:
                models.append(parts[0])

        return models

    def pull_model(self, model_name: str) -> tuple[bool, str]:
        if not self.is_available():
            return False, "Ollama CLI не знайдено."

        result = subprocess.run(
            ["ollama", "pull", model_name],
            text=True,
            capture_output=True,
            check=False,
        )

        output = "\n".join(
            part.strip()
            for part in [result.stdout, result.stderr]
            if part.strip()
        )

        if result.returncode != 0:
            return False, output or f"Не вдалося скачати модель: {model_name}"

        return True, output or f"Модель скачана: {model_name}"