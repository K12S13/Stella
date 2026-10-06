from __future__ import annotations

import time
from pathlib import Path

from stella.core.config import StellaConfig
from stella.stt.providers.base import STTResult
from stella.stt.providers.whisper_provider import WhisperProvider


def transcribe_wav_with_whisper(
    config: StellaConfig,
    wav_path: str | Path,
) -> STTResult:
    try:
        provider = WhisperProvider(
            model_name=config.stt_whisper_model,
            sample_rate=config.stt_sample_rate,
            device_index=config.stt_device_index,
            device=config.stt_whisper_device,
            compute_type=config.stt_whisper_compute_type,
            language=config.stt_whisper_language,
            beam_size=config.stt_whisper_beam_size,
            vad_filter=config.stt_whisper_vad_filter,
        )

        model = provider._load_model()

        started = time.monotonic()

        segments, info = model.transcribe(
            str(wav_path),
            language=config.stt_whisper_language or None,
            task="transcribe",
            beam_size=config.stt_whisper_beam_size,
            vad_filter=config.stt_whisper_vad_filter,
            condition_on_previous_text=False,
            temperature=0.0,
            initial_prompt=(
                "Це українська або російська голосова команда для desktop assistant Stella. "
                "Розпізнай текст без перекладу."
            ),
        )

        parts: list[str] = []

        for segment in segments:
            text = segment.text.strip()

            if text:
                parts.append(text)

        final_text = " ".join(parts).strip()
        elapsed = time.monotonic() - started

        if not final_text:
            return STTResult(
                success=False,
                text="",
                error=f"Whisper нічого не розпізнав. Processing: {elapsed:.2f}s",
                provider="whisper",
            )

        return STTResult(
            success=True,
            text=final_text,
            error=f"Processing: {elapsed:.2f}s",
            provider="whisper",
        )

    except Exception as error:
        return STTResult(
            success=False,
            text="",
            error=str(error),
            provider="whisper",
        )