from __future__ import annotations
from stella.text.response_normalizer import normalize_response_text, sanitize_stream_chunk

import re

from PySide6.QtCore import QObject, Signal

from stella.core.config import StellaConfig
from stella.llm.ollama_provider import OllamaProvider


class LLMStreamWorker(QObject):
    chunk = Signal(str)
    speak_chunk = Signal(str)
    finished = Signal(str, str, str)
    failed = Signal(str)

    def __init__(self, config: StellaConfig, text: str) -> None:
        super().__init__()
        self.config = config
        self.text = text

    def run(self) -> None:
        provider = OllamaProvider(self.config)

        full_text_parts: list[str] = []
        speech_buffer = ""

        try:
            for chunk in provider.stream(self.text):
                clean_chunk = sanitize_stream_chunk(chunk)

                if not clean_chunk:
                    continue

                full_text_parts.append(clean_chunk)
                speech_buffer += clean_chunk

                self.chunk.emit(clean_chunk)

                sentences, speech_buffer = self._extract_speakable_sentences(
                    speech_buffer
                )

                for sentence in sentences:
                    normalized_sentence = normalize_response_text(sentence)

                    if normalized_sentence:
                        self.speak_chunk.emit(normalized_sentence)

            remaining = speech_buffer.strip()

            if remaining:
                normalized_remaining = normalize_response_text(remaining)

                if normalized_remaining:
                    self.speak_chunk.emit(normalized_remaining)

            full_text = "".join(full_text_parts).strip()

            self.finished.emit(
                full_text,
                "ollama",
                self.config.local_model,
            )

        except Exception as error:
            self.failed.emit(str(error))

    def _extract_speakable_sentences(self, text: str) -> tuple[list[str], str]:
        normalized = text.replace("\n", " ").strip()

        if not normalized:
            return [], ""

        sentences: list[str] = []

        while True:
            # Emit sentence immediately after punctuation.
            # Do not wait for the next token/space.
            match = re.search(r"^(.+?[.!?…])(?:\s+|$)", normalized)

            if not match:
                break

            sentence = match.group(1).strip()
            normalized = normalized[match.end():].strip()

            if len(sentence) >= 12:
                sentences.append(sentence)

        # Safety fallback: long sentence without punctuation.
        if len(normalized) >= 180:
            sentences.append(normalized[:180].strip())
            normalized = normalized[180:].strip()

        return sentences, normalized