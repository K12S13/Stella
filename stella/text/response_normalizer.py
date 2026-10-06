from __future__ import annotations

import re


CJK_RE = re.compile(
    r"[\u3400-\u4DBF\u4E00-\u9FFF\u3040-\u30FF\uAC00-\uD7AF]"
)

ZERO_WIDTH_RE = re.compile(r"[\u200B-\u200F\u202A-\u202E\u2060-\u206F]")


LATIN_TO_CYRILLIC_CONFUSABLES = {
    "A": "А",
    "B": "В",
    "C": "С",
    "E": "Е",
    "H": "Н",
    "I": "І",
    "K": "К",
    "M": "М",
    "O": "О",
    "P": "Р",
    "T": "Т",
    "X": "Х",
    "Y": "У",
    "a": "а",
    "c": "с",
    "e": "е",
    "i": "і",
    "o": "о",
    "p": "р",
    "x": "х",
    "y": "у",
}


CYRILLIC_TO_LATIN_CONFUSABLES = {
    "А": "A",
    "В": "B",
    "С": "C",
    "Е": "E",
    "Н": "H",
    "І": "I",
    "К": "K",
    "М": "M",
    "О": "O",
    "Р": "P",
    "Т": "T",
    "Х": "X",
    "а": "a",
    "е": "e",
    "о": "o",
    "р": "p",
    "с": "c",
    "х": "x",
}


def sanitize_stream_chunk(text: str) -> str:
    """
    Safe lightweight cleanup for live streaming chunks.
    Does not do heavy token correction because chunks may split words.
    """
    cleaned = ZERO_WIDTH_RE.sub("", text)
    cleaned = CJK_RE.sub("", cleaned)
    return cleaned


def normalize_response_text(text: str) -> str:
    """
    Full cleanup for completed phrases/sentences before TTS and final display.
    Fixes:
    - Chinese/Japanese/Korean symbol garbage;
    - Latin letters inside Cyrillic words: привeт -> привет;
    - Cyrillic lookalikes inside English tokens: СPU -> CPU;
    - weird invisible Unicode;
    - broken punctuation spacing.
    """
    cleaned = text.strip()

    if not cleaned:
        return ""

    cleaned = ZERO_WIDTH_RE.sub("", cleaned)
    cleaned = CJK_RE.sub("", cleaned)

    tokens = re.findall(
        r"[A-Za-zА-Яа-яІіЇїЄєҐґЁё0-9_+#./:\-]+|[^\w\s]+|\s+",
        cleaned,
        flags=re.UNICODE,
    )

    fixed_tokens: list[str] = []

    for token in tokens:
        if not token.strip():
            fixed_tokens.append(" ")
            continue

        if _is_word_like(token):
            fixed_tokens.append(_fix_mixed_script_token(token))
        else:
            fixed_tokens.append(token)

    result = "".join(fixed_tokens)

    result = result.replace("ʼ", "'").replace("’", "'").replace("`", "'")
    result = result.replace("“", '"').replace("”", '"')
    result = result.replace("«", '"').replace("»", '"')
    result = result.replace("—", "-").replace("–", "-")
    result = result.replace("…", "...")

    result = re.sub(r"\s+", " ", result)
    result = re.sub(r"\s+([,.!?;:])", r"\1", result)
    result = re.sub(r"([,.!?;:])([^\s])", r"\1 \2", result)

    return result.strip()


def normalize_for_tts(text: str) -> str:
    """
    TTS-specific cleanup. Keeps English technical terms,
    but removes garbage that voices pronounce badly.
    """
    cleaned = normalize_response_text(text)

    # Remove repeated accidental punctuation.
    cleaned = re.sub(r"([!?.,])\1{2,}", r"\1", cleaned)

    # Avoid reading markdown bullets/styles too literally.
    cleaned = cleaned.replace("**", "")
    cleaned = cleaned.replace("__", "")
    cleaned = cleaned.replace("`", "")

    return cleaned.strip()


def _is_word_like(token: str) -> bool:
    return bool(
        re.search(
            r"[A-Za-zА-Яа-яІіЇїЄєҐґЁё]",
            token,
            flags=re.UNICODE,
        )
    )


def _fix_mixed_script_token(token: str) -> str:
    latin_count = len(re.findall(r"[A-Za-z]", token))
    cyrillic_count = len(re.findall(r"[А-Яа-яІіЇїЄєҐґЁё]", token))

    if latin_count == 0 or cyrillic_count == 0:
        return token

    # Ukrainian/Russian word polluted by Latin letters:
    # процeсор -> процесор
    if cyrillic_count >= latin_count:
        return "".join(
            LATIN_TO_CYRILLIC_CONFUSABLES.get(char, char)
            for char in token
        )

    # English token polluted by Cyrillic lookalikes:
    # СPU -> CPU
    return "".join(
        CYRILLIC_TO_LATIN_CONFUSABLES.get(char, char)
        for char in token
    )