from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher

from stella.commands.registry import Command, CommandRegistry
from stella.core.config import StellaConfig


@dataclass
class MatchResult:
    matched: bool
    source: str
    confidence: float
    command: Command | None = None
    dynamic_action: dict | None = None
    original_text: str = ""
    normalized_text: str = ""


def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = text.replace("ё", "е")
    text = re.sub(r"[^\w\sа-яіїєґ'-]", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def strip_assistant_name(text: str, assistant_name: str) -> str:
    text = normalize_text(text)
    assistant_name = normalize_text(assistant_name)

    if text.startswith(assistant_name + " "):
        return text[len(assistant_name):].strip()

    return text


def similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


class CommandMatcher:
    def __init__(self, registry: CommandRegistry, config: StellaConfig) -> None:
        self.registry = registry
        self.config = config

    def match(self, text: str) -> MatchResult:
        original = text
        normalized = normalize_text(text)
        without_name = strip_assistant_name(normalized, self.config.assistant_name)

        exact_match = self._match_exact(normalized, without_name)
        if exact_match:
            return MatchResult(
                matched=True,
                source="exact_phrase",
                confidence=1.0,
                command=exact_match,
                original_text=original,
                normalized_text=without_name,
            )

        fuzzy_match, fuzzy_score = self._match_fuzzy(without_name)
        if fuzzy_match and fuzzy_score >= 0.86:
            return MatchResult(
                matched=True,
                source="fuzzy_phrase",
                confidence=fuzzy_score,
                command=fuzzy_match,
                original_text=original,
                normalized_text=without_name,
            )

        dynamic = self._match_dynamic_app_action(without_name)
        if dynamic:
            return MatchResult(
                matched=True,
                source="dynamic_app_action",
                confidence=0.78,
                dynamic_action=dynamic,
                original_text=original,
                normalized_text=without_name,
            )

        return MatchResult(
            matched=False,
            source="none",
            confidence=0.0,
            original_text=original,
            normalized_text=without_name,
        )

    def _match_exact(self, normalized: str, without_name: str) -> Command | None:
        for command in self.registry.all():
            for phrase in command.phrases:
                phrase = normalize_text(phrase)
                phrase_without_name = strip_assistant_name(phrase, self.config.assistant_name)

                if normalized == phrase or without_name == phrase_without_name:
                    return command

        return None

    def _match_fuzzy(self, text: str) -> tuple[Command | None, float]:
        best_command: Command | None = None
        best_score = 0.0

        for command in self.registry.all():
            for phrase in command.phrases:
                phrase = strip_assistant_name(phrase, self.config.assistant_name)
                score = similarity(text, phrase)

                if score > best_score:
                    best_score = score
                    best_command = command

        return best_command, best_score

    def _match_dynamic_app_action(self, text: str) -> dict | None:
        open_patterns = [
            r"^(відкрий|запусти|відкрити|запустить|открой|запусти|open|launch)\s+(.+)$",
        ]

        close_patterns = [
            r"^(закрий|закрити|закрой|close|kill)\s+(.+)$",
        ]

        for pattern in open_patterns:
            match = re.match(pattern, text)
            if match:
                app_name = match.group(2).strip()
                return {
                    "type": "open_app",
                    "app": app_name,
                    "dangerous": False,
                    "needs_learning_offer": True,
                }

        for pattern in close_patterns:
            match = re.match(pattern, text)
            if match:
                app_name = match.group(2).strip()
                return {
                    "type": "close_app",
                    "app": app_name,
                    "dangerous": True,
                    "needs_learning_offer": False,
                }

        return None