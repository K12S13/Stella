from __future__ import annotations

import threading

from stella.core.config import StellaConfig
from stella.stt.providers.base import STTResult
from stella.stt.providers.dummy_provider import DummySTTProvider



class STTRouter:
    def __init__(self, config: StellaConfig) -> None:
        self.config = config

    def listen_once(
        self,
        stop_event: threading.Event | None = None,
    ) -> STTResult:
        provider = self.config.stt_provider

        if provider == "dummy":
            return DummySTTProvider().listen_once(
                seconds=self.config.stt_listen_seconds,
                stop_event=stop_event,
            )

        if provider == "vosk":
            try:
                from stella.stt.providers.vosk_provider import VoskProvider
            except ModuleNotFoundError as error:
                return STTResult(
                    success=False,
                    text="",
                    error=(
                        f"Vosk dependency missing: {error}. "
                        "Run: python -m pip install sounddevice vosk numpy"
                    ),
                    provider="vosk",
                )

            return VoskProvider(
                model_path=self.config.stt_vosk_model_path,
                sample_rate=self.config.stt_sample_rate,
                device_index=self.config.stt_device_index,
            ).listen_once(
                seconds=self.config.stt_listen_seconds,
                stop_event=stop_event,
            )

        if provider == "whisper":
            try:
                from stella.stt.providers.whisper_provider import WhisperProvider
            except ModuleNotFoundError as error:
                return STTResult(
                    success=False,
                    text="",
                    error=(
                        f"Whisper dependency missing: {error}. "
                        "Run: python -m pip install faster-whisper sounddevice numpy"
                    ),
                    provider="whisper",
                )

            return WhisperProvider(
                model_name=self.config.stt_whisper_model,
                sample_rate=self.config.stt_sample_rate,
                device_index=self.config.stt_device_index,
                device=self.config.stt_whisper_device,
                compute_type=self.config.stt_whisper_compute_type,
                language=self.config.stt_whisper_language,
                beam_size=self.config.stt_whisper_beam_size,
                vad_filter=self.config.stt_whisper_vad_filter,
            ).listen_once(
                seconds=self.config.stt_listen_seconds,
                stop_event=stop_event,
            )

        if provider == "none":
            return STTResult(
                success=False,
                text="",
                error="STT provider вимкнено. Постав stt.provider: whisper, vosk або dummy.",
                provider="none",
            )

        return STTResult(
            success=False,
            text="",
            error=f"STT provider поки не підтримується: {provider}",
            provider=provider,
        )