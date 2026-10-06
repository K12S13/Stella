from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

import yaml


BASE_WORDS = {
    # assistant
    "стелла", "stella",

    # Ukrainian commands
    "відкрий", "відкрити", "закрий", "закрити", "запусти",
    "поясни", "скажи", "знайди", "покажи", "напиши",
    "збільш", "зменш", "додай", "прибери",
    "звук", "гучність", "мікрофон", "браузер", "файрфокс",
    "дискорд", "телеграм", "налаштування",

    # Ukrainian common
    "що", "таке", "як", "це", "мені", "треба", "потрібно",
    "коротко", "детально", "швидко", "будь", "ласка",
    "кеш", "процесора", "процесор", "память", "памʼять", "пам'ять",

    # Russian commands
    "открой", "закрой", "запусти", "объясни", "скажи",
    "найди", "покажи", "напиши", "увеличь", "уменьши",
    "звук", "громкость", "микрофон", "браузер",
    "дискорд", "телеграм", "настройки",

    # Russian common
    "что", "такое", "как", "это", "мне", "надо", "нужно",
    "коротко", "подробно", "быстро", "пожалуйста",
    "кэш", "процессора", "процессор", "память",

    # English technical
    "cpu", "gpu", "cache", "browser", "github", "python",
    "linux", "windows", "ollama", "whisper", "piper",
}


SHORT_WORDS = {"я", "і", "й", "в", "у", "з", "a", "i"}


def cleanup_stt_text(text: str) -> str:
    normalized = _normalize_text(text)
    words = _vocabulary()

    tokens = re.findall(
        r"[A-Za-zА-Яа-яІіЇїЄєҐґЁёЫыЭэЪъʼ'’]+|\d+|[^\w\s]+|\s+",
        normalized,
        flags=re.UNICODE,
    )

    result: list[str] = []

    for token in tokens:
        if not token.strip():
            result.append(" ")
            continue

        if _is_word_token(token):
            result.append(_split_glued_word(token, words))
        else:
            result.append(token)

    cleaned = "".join(result)

    cleaned = re.sub(r"\s+", " ", cleaned)
    cleaned = re.sub(r"\s+([,.!?;:])", r"\1", cleaned)
    cleaned = re.sub(r"([,.!?;:])([^\s])", r"\1 \2", cleaned)

    return cleaned.strip()


def _normalize_text(text: str) -> str:
    replacements = {
        "’": "'",
        "ʼ": "'",
        "`": "'",
        "ё": "е",
        "Ё": "Е",
    }

    normalized = text.strip().lower()

    for old, new in replacements.items():
        normalized = normalized.replace(old, new)

    return normalized


def _is_word_token(token: str) -> bool:
    return bool(
        re.fullmatch(
            r"[A-Za-zА-Яа-яІіЇїЄєҐґЁёЫыЭэЪъʼ'’]+",
            token,
            flags=re.UNICODE,
        )
    )


@lru_cache(maxsize=1)
def _vocabulary() -> frozenset[str]:
    words = set(BASE_WORDS)

    commands_path = Path("stella/data/commands.yaml")

    if commands_path.exists():
        try:
            data = yaml.safe_load(commands_path.read_text(encoding="utf-8")) or {}

            for command in data.get("commands", []):
                for phrase in command.get("phrases", []):
                    for word in _extract_words(str(phrase)):
                        words.add(word)

                raw_command = command.get("command")

                if raw_command:
                    for word in _extract_words(str(raw_command)):
                        words.add(word)

        except Exception:
            pass

    return frozenset(words)


def _extract_words(text: str) -> list[str]:
    return [
        word.lower()
        for word in re.findall(
            r"[A-Za-zА-Яа-яІіЇїЄєҐґЁёЫыЭэЪъʼ'’]+",
            text,
            flags=re.UNICODE,
        )
    ]


def _split_glued_word(token: str, vocabulary: frozenset[str]) -> str:
    token = token.lower()

    if len(token) < 8:
        return token

    if token in vocabulary:
        return token

    pieces = _word_break(token, vocabulary)

    if not pieces:
        return token

    if len(pieces) <= 1:
        return token

    # Safety: do not create garbage like one-letter spam.
    small_pieces = [
        piece
        for piece in pieces
        if len(piece) <= 1 and piece not in SHORT_WORDS
    ]

    if small_pieces:
        return token

    return " ".join(pieces)


def _word_break(
    token: str,
    vocabulary: frozenset[str],
) -> list[str] | None:
    n = len(token)
    dp: list[list[str] | None] = [None] * (n + 1)
    dp[0] = []

    for i in range(n):
        if dp[i] is None:
            continue

        for j in range(i + 1, n + 1):
            piece = token[i:j]

            if piece not in vocabulary:
                continue

            if len(piece) <= 1 and piece not in SHORT_WORDS:
                continue

            candidate = [*dp[i], piece]

            if dp[j] is None or _candidate_score(candidate) > _candidate_score(dp[j]):
                dp[j] = candidate

    return dp[n]


def _candidate_score(parts: list[str]) -> int:
    # Prefer fewer, longer words.
    return sum(len(part) * len(part) for part in parts) - len(parts) * 3
