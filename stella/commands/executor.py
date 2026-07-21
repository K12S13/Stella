from __future__ import annotations

import shutil
import subprocess
import time
from dataclasses import dataclass

from stella.commands.matcher import MatchResult
from stella.commands.registry import Command
from stella.core.config import StellaConfig


@dataclass
class ExecutionResult:
    success: bool
    message: str
    requires_confirmation: bool = False
    pending_action: dict | None = None


APP_ALIASES: dict[str, list[str]] = {
    "firefox": ["firefox"],
    "браузер": ["firefox"],
    "browser": ["firefox"],

    "discord": ["discord"],
    "діскорд": ["discord"],
    "дискорд": ["discord"],

    "telegram": ["telegram-desktop", "telegram"],
    "телеграм": ["telegram-desktop", "telegram"],

    "pycharm": ["pycharm", "pycharm-community"],
    "пайчарм": ["pycharm", "pycharm-community"],

    "steam": ["steam"],
    "стім": ["steam"],

    "terminal": ["konsole", "gnome-terminal", "kitty", "alacritty"],
    "термінал": ["konsole", "gnome-terminal", "kitty", "alacritty"],
}


class CommandExecutor:
    def __init__(self, config: StellaConfig) -> None:
        self.config = config

    def execute_match(self, match: MatchResult, confirmed: bool = False) -> ExecutionResult:
        if not match.matched:
            return ExecutionResult(
                success=False,
                message=f"Не знайшла команду для: {match.original_text}",
            )

        if match.command:
            return self._execute_registered_command(match.command, confirmed=confirmed)

        if match.dynamic_action:
            return self._execute_dynamic_action(match.dynamic_action, confirmed=confirmed)

        return ExecutionResult(
            success=False,
            message="Команду розпізнано, але немає дії для виконання.",
        )

    def _execute_registered_command(self, command: Command, confirmed: bool = False) -> ExecutionResult:
        if command.dangerous and self.config.confirm_dangerous_actions and not confirmed:
            return ExecutionResult(
                success=False,
                message=f"Команда '{command.id}' може бути небезпечною. Підтверди повторно.",
                requires_confirmation=True,
            )

        if command.type == "open_app":
            if not command.app:
                return ExecutionResult(False, f"У команді '{command.id}' не вказано app.")
            return self._open_app(command.app)

        if command.type == "shell":
            if not self.config.allow_shell_commands:
                return ExecutionResult(False, "Shell-команди вимкнені в налаштуваннях.")
            if not command.command:
                return ExecutionResult(False, f"У команді '{command.id}' не вказано shell command.")
            return self._run_shell(command.command)

        if command.type == "close_app":
            if not command.app:
                return ExecutionResult(False, f"У команді '{command.id}' не вказано app.")
            return self._close_app(command.app)

        return ExecutionResult(False, f"Тип команди поки не підтримується: {command.type}")

    def _execute_dynamic_action(self, action: dict, confirmed: bool = False) -> ExecutionResult:
        action_type = action.get("type")
        app = str(action.get("app", "")).strip()

        if not app:
            return ExecutionResult(False, "Не вказано назву програми.")

        if action.get("dangerous") and self.config.confirm_dangerous_actions and not confirmed:
            return ExecutionResult(
                success=False,
                message=f"Дія '{action_type} {app}' може закрити програму або втратити незбережені дані. Підтверди повторно.",
                requires_confirmation=True,
            )

        if action_type == "open_app":
            return self._open_app(app)

        if action_type == "close_app":
            return self._close_app(app)

        return ExecutionResult(False, f"Динамічна дія поки не підтримується: {action_type}")

    def _resolve_app_command(self, app_name: str) -> str | None:
        app_name = app_name.lower().strip()
        candidates = APP_ALIASES.get(app_name, [app_name])

        for candidate in candidates:
            resolved = shutil.which(candidate)
            if resolved:
                return resolved

        return None

    def _open_app(self, app_name: str) -> ExecutionResult:
        resolved = self._resolve_app_command(app_name)

        if not resolved:
            return ExecutionResult(
                success=False,
                message=f"Не знайшла програму '{app_name}' у PATH. Її треба додати в aliases або встановити.",
            )

        try:
            subprocess.Popen(
                [resolved],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
            return ExecutionResult(True, f"Готово, відкриваю {app_name}.")
        except Exception as error:
            return ExecutionResult(False, f"Не вдалося відкрити {app_name}: {error}")

    def _close_app(self, app_name: str) -> ExecutionResult:
        normalized_app = app_name.lower().strip()

        close_patterns = {
            "discord": ".config/discord",
            "діскорд": ".config/discord",
            "дискорд": ".config/discord",

            "firefox": "firefox",
            "браузер": "firefox",

            "telegram": "telegram-desktop",
            "телеграм": "telegram-desktop",
        }

        pattern = close_patterns.get(normalized_app)

        if not pattern:
            aliases = APP_ALIASES.get(normalized_app, [app_name])
            pattern = aliases[0]

        # 1. Перевіряємо, чи процес існує.
        before = subprocess.run(
            ["pgrep", "-afi", pattern],
            text=True,
            capture_output=True,
            check=False,
        )

        if before.returncode != 0 or not before.stdout.strip():
            return ExecutionResult(
                False,
                f"Не знайшла процес для {app_name}. Pattern: {pattern}"
            )

        # 2. М'яке закриття.
        subprocess.run(
            ["pkill", "-TERM", "-f", pattern],
            text=True,
            capture_output=True,
            check=False,
        )

        time.sleep(1.0)

        after_term = subprocess.run(
            ["pgrep", "-afi", pattern],
            text=True,
            capture_output=True,
            check=False,
        )

        if after_term.returncode != 0 or not after_term.stdout.strip():
            return ExecutionResult(True, f"Закрила {app_name}.")

        # 3. Примусове закриття.
        subprocess.run(
            ["pkill", "-KILL", "-f", pattern],
            text=True,
            capture_output=True,
            check=False,
        )

        time.sleep(0.5)

        after_kill = subprocess.run(
            ["pgrep", "-afi", pattern],
            text=True,
            capture_output=True,
            check=False,
        )

        if after_kill.returncode != 0 or not after_kill.stdout.strip():
            return ExecutionResult(True, f"Примусово закрила {app_name}.")

        return ExecutionResult(
            False,
            f"Спробувала закрити {app_name}, але процес ще активний:\n{after_kill.stdout.strip()}"
        )

    def _run_shell(self, command: str) -> ExecutionResult:
        try:
            completed = subprocess.run(
                command,
                shell=True,
                text=True,
                capture_output=True,
                check=False,
            )

            output = completed.stdout.strip()
            error = completed.stderr.strip()

            if completed.returncode == 0:
                if output:
                    return ExecutionResult(True, f"Команда виконана:\n{output}")
                return ExecutionResult(True, "Команда виконана.")

            return ExecutionResult(
                False,
                f"Команда завершилась з кодом {completed.returncode}:\n{error or output}",
            )
        except Exception as error:
            return ExecutionResult(False, f"Не вдалося виконати shell-команду: {error}")