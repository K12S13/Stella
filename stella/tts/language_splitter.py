from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class TextChunk:
    language: str
    text: str


UKRAINIAN_ONLY_CHARS = set("іїєґІЇЄҐ")
RUSSIAN_ONLY_CHARS = set("ыэъёЫЭЪЁ")


UKRAINIAN_WORDS = {
    "це", "я", "ти", "він", "вона", "воно", "ми", "ви", "вони",
    "що", "як", "якщо", "щоб", "або", "але", "та", "і", "й",
    "мене", "тебе", "його", "її", "нам", "вам",
    "треба", "потрібно", "можна", "немає", "є",
    "зараз", "вже", "дуже", "тут", "там",
    "відкрий", "закрий", "запусти", "поясни", "скажи",
    "коротко", "налаштування", "браузер", "звук", "гучність",
    "працює", "памʼять", "пам'ять", "швидка", "поруч",
}


RUSSIAN_WORDS = {
    "это", "я", "ты", "он", "она", "оно", "мы", "вы", "они",
    "что", "как", "если", "чтобы", "или", "но", "и",
    "меня", "тебя", "его", "ее", "нам", "вам",
    "нужно", "надо", "можно", "нет", "есть",
    "сейчас", "уже", "очень", "тут", "там",
    "открой", "закрой", "запусти", "объясни", "скажи",
    "коротко", "настройки", "браузер", "звук", "громкость",
    "работает", "память", "быстрая", "рядом",
}


def split_text_by_language(
    text: str,
    default_language: str = "uk",
) -> list[TextChunk]:
    sentences = _split_sentences(text)

    chunks: list[TextChunk] = []

    for sentence in sentences:
        if not sentence.strip():
            continue

        base_language = detect_sentence_language(sentence, default_language)
        sentence_chunks = _split_latin_islands(sentence, base_language)

        chunks.extend(sentence_chunks)

    return _merge_neighbor_chunks(chunks)


def detect_sentence_language(
    text: str,
    default_language: str = "uk",
) -> str:
    stripped = text.strip()

    if not stripped:
        return default_language

    latin_count = len(re.findall(r"[A-Za-z]", stripped))
    cyrillic_count = len(re.findall(r"[А-Яа-яІіЇїЄєҐґЁёЫыЭэЪъ]", stripped))

    if latin_count > 0 and latin_count >= cyrillic_count:
        return "en"

    uk_score = 0
    ru_score = 0

    for char in stripped:
        if char in UKRAINIAN_ONLY_CHARS:
            uk_score += 5

        if char in RUSSIAN_ONLY_CHARS:
            ru_score += 5

    words = [
        word.lower()
        for word in re.findall(
            r"[А-Яа-яІіЇїЄєҐґЁёЫыЭэЪъʼ'’]+",
            stripped,
            flags=re.UNICODE,
        )
    ]

    for word in words:
        if word in UKRAINIAN_WORDS:
            uk_score += 2

        if word in RUSSIAN_WORDS:
            ru_score += 2

    if uk_score > ru_score:
        return "uk"

    if ru_score > uk_score:
        return "ru"

    return default_language


def _split_sentences(text: str) -> list[str]:
    parts = re.findall(r"[^.!?…\n]+[.!?…]*|\n+", text)

    result: list[str] = []

    for part in parts:
        if part == "\n":
            continue

        if part.strip():
            result.append(part.strip())

    return result


def _split_latin_islands(
    sentence: str,
    base_language: str,
) -> list[TextChunk]:
    if base_language == "en":
        return [TextChunk("en", sentence.strip())]

    # Latin islands like "CPU cache", "GitHub", "large-v3-turbo"
    pattern = re.compile(
        r"[A-Za-z][A-Za-z0-9_+#./:\-]*(?:\s+[A-Za-z][A-Za-z0-9_+#./:\-]*)*"
    )

    chunks: list[TextChunk] = []
    last_end = 0

    for match in pattern.finditer(sentence):
        before = sentence[last_end:match.start()].strip()

        if before:
            chunks.append(TextChunk(base_language, before))

        latin_text = match.group(0).strip()

        if latin_text:
            chunks.append(TextChunk("en", latin_text))

        last_end = match.end()

    after = sentence[last_end:].strip()

    if after:
        chunks.append(TextChunk(base_language, after))

    if not chunks:
        return [TextChunk(base_language, sentence.strip())]

    return chunks


def _merge_neighbor_chunks(chunks: list[TextChunk]) -> list[TextChunk]:
    merged: list[TextChunk] = []

    for chunk in chunks:
        clean_text = chunk.text.strip()

        if not clean_text:
            continue

        if not merged:
            merged.append(TextChunk(chunk.language, clean_text))
            continue

        previous = merged[-1]

        if previous.language == chunk.language:
            merged[-1] = TextChunk(
                previous.language,
                f"{previous.text} {clean_text}".strip(),
            )
        else:
            merged.append(TextChunk(chunk.language, clean_text))

    return merged