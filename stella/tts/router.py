from __future__ import annotations

from dataclasses import replace

from stella.core.config import StellaConfig
from stella.tts.language_splitter import split_text_by_language
from stella.tts.piper_provider import PiperTTSProvider, TTSResult
from stella.text.response_normalizer import normalize_for_tts


class TTSRouter:
    def __init__(self, config: StellaConfig) -> None:
        self.config = config

    def speak(self, text: str) -> TTSResult:
        if not self.config.tts_enabled:
            return TTSResult(False, "TTS вимкнено.")

        text = normalize_for_tts(text)

        if not text:
            return TTSResult(False, "TTS text became empty after normalization.")

        if not self.config.tts_language_routing:
            return self._speak_with_provider(
                text=text,
                provider=self.config.tts_provider,
                model_path=self.config.tts_piper_model,
            )

        chunks = split_text_by_language(
            text,
            default_language=self.config.tts_default_language,
        )

        if not chunks:
            return TTSResult(False, "No TTS chunks generated.")

        errors: list[str] = []

        for chunk in chunks:
            provider = self._provider_for_language(chunk.language)
            model_path = self._model_for_language(chunk.language)

            result = self._speak_with_provider(
                text=chunk.text,
                provider=provider,
                model_path=model_path,
            )

            if not result.success:
                errors.append(f"{chunk.language}: {result.message}")

        if errors:
            return TTSResult(False, "\n".join(errors))

        return TTSResult(True, "TTS OK.")

    def _provider_for_language(self, language: str) -> str:
        if language == "uk":
            return self.config.tts_uk_provider

        if language == "en":
            return self.config.tts_en_provider

        if language == "ru":
            return self.config.tts_ru_provider

        return self.config.tts_provider

    def _model_for_language(self, language: str) -> str:
        if language == "uk":
            return self.config.tts_uk_model

        if language == "en":
            return self.config.tts_en_model

        if language == "ru":
            return self.config.tts_ru_model

        return self.config.tts_piper_model

    def _speak_with_provider(
        self,
        text: str,
        provider: str,
        model_path: str,
    ) -> TTSResult:
        if provider == "piper":
            voice_config = replace(
                self.config,
                tts_piper_model=model_path,
            )
            return PiperTTSProvider(voice_config).speak(text, model_path=model_path)

        return TTSResult(
            False,
            (
                f"TTS provider '{provider}' ще не реалізований у коді. "
                "Зараз стабільно працює provider: piper."
            ),
        )