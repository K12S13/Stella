from __future__ import annotations

from stella.commands.executor import CommandExecutor
from stella.commands.matcher import CommandMatcher
from stella.commands.registry import CommandRegistry
from stella.core.config import load_config
from stella.core.history import HistoryLogger


def stella_print(history: HistoryLogger, message: str) -> None:
    print(f"Stella> {message}")
    history.write("Stella", message)


def main() -> None:
    config = load_config()
    registry = CommandRegistry.from_yaml()
    matcher = CommandMatcher(registry, config)
    executor = CommandExecutor(config)
    history = HistoryLogger()

    print("Stella Core v0.1")
    print(f"Assistant name: {config.assistant_name}")
    print("Напиши команду. Для виходу: exit / quit / вихід")
    print("Історія зберігається в: stella/data/history.log")
    print()

    history.write("System", "Stella Core started")

    pending_match = None

    while True:
        try:
            text = input("you> ").strip()
        except KeyboardInterrupt:
            print("\nВихід.")
            history.write("System", "Stella Core stopped by KeyboardInterrupt")
            break

        if not text:
            continue

        history.write("User", text)

        if text.lower() in {"exit", "quit", "вихід", "вийти"}:
            stella_print(history, "Вихід.")
            break

        if pending_match is not None:
            confirmation = text.lower().strip()

            if confirmation in {"так", "yes", "y", "підтверджую", "confirm", "виконуй"}:
                result = executor.execute_match(pending_match, confirmed=True)
                pending_match = None
                stella_print(history, result.message)
                continue

            if confirmation in {"ні", "no", "n", "скасувати", "cancel"}:
                pending_match = None
                stella_print(history, "Скасовано.")
                continue

            stella_print(history, "Очікую підтвердження: так / ні")
            continue

        match = matcher.match(text)

        debug_message = (
            f"source={match.source}, "
            f"confidence={match.confidence:.2f}, "
            f"normalized='{match.normalized_text}'"
        )

        print(f"[debug] {debug_message}")
        history.write("Debug", debug_message)

        result = executor.execute_match(match)

        if result.requires_confirmation:
            pending_match = match
            stella_print(history, result.message)
            stella_print(history, "Напиши: так / ні")
            continue

        stella_print(history, result.message)


if __name__ == "__main__":
    main()