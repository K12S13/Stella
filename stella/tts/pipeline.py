from __future__ import annotations

import queue
import threading
from dataclasses import dataclass, replace
from pathlib import Path

from stella.core.config import StellaConfig
from stella.tts.audio_player import AudioPlayer
from stella.tts.edge_provider import EdgeTTSProvider
from stella.tts.language_splitter import split_text_by_language
from stella.tts.piper_provider import PiperTTSProvider
from stella.text.response_normalizer import normalize_for_tts


@dataclass(frozen=True)
class SpeechJob:
    text: str
    language: str
    provider: str
    model_path: str


@dataclass(frozen=True)
class PlayJob:
    audio_path: str


class TTSPipeline:
    def __init__(self, config: StellaConfig) -> None:
        self.config = config

        self._synth_queue: queue.Queue[SpeechJob | None] = queue.Queue()
        self._play_queue: queue.Queue[PlayJob | None] = queue.Queue()

        self._stop_event = threading.Event()

        self._synth_thread: threading.Thread | None = None
        self._play_thread: threading.Thread | None = None

    def start(self) -> None:
        if self._synth_thread is not None and self._synth_thread.is_alive():
            return

        self._stop_event.clear()

        self._synth_thread = threading.Thread(
            target=self._synth_loop,
            name="StellaTTSSynth",
            daemon=True,
        )

        self._play_thread = threading.Thread(
            target=self._play_loop,
            name="StellaTTSPlay",
            daemon=True,
        )

        self._synth_thread.start()
        self._play_thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        self._synth_queue.put(None)
        self._play_queue.put(None)

    def clear(self) -> None:
        self._drain_queue(self._synth_queue)
        self._drain_queue(self._play_queue)

    def enqueue(self, text: str, source: str = "llm") -> None:
        clean_text = normalize_for_tts(text)

        if not clean_text:
            return

        if not self.config.tts_enabled:
            return

        if source == "llm" and not self.config.tts_speak_llm_answers:
            return

        if source == "command" and not self.config.tts_speak_command_results:
            return

        self.start()

        if self.config.tts_language_routing:
            chunks = split_text_by_language(
                clean_text,
                default_language=self.config.tts_default_language,
            )
        else:
            chunks = []

        if not chunks:
            self._synth_queue.put(
                SpeechJob(
                    text=clean_text,
                    language=self.config.tts_default_language,
                    provider=self.config.tts_provider,
                    model_path=self.config.tts_piper_model,
                )
            )
            return

        for chunk in chunks:
            self._synth_queue.put(
                SpeechJob(
                    text=chunk.text,
                    language=chunk.language,
                    provider=self._provider_for_language(chunk.language),
                    model_path=self._model_for_language(chunk.language),
                )
            )

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

    def _edge_voice_for_language(self, language: str) -> str:
        if language == "uk":
            return self.config.tts_edge_uk_voice

        if language == "en":
            return self.config.tts_edge_en_voice

        if language == "ru":
            return self.config.tts_edge_ru_voice

        return self.config.tts_edge_uk_voice

    def _synth_loop(self) -> None:
        while not self._stop_event.is_set():
            job = self._synth_queue.get()

            if job is None:
                break

            audio_path = self._synthesize(job)

            if audio_path is None and job.provider != self.config.tts_fallback_provider:
                fallback_job = replace(
                    job,
                    provider=self.config.tts_fallback_provider,
                )
                audio_path = self._synthesize(fallback_job)

            if audio_path is not None:
                self._play_queue.put(PlayJob(audio_path))

    def _synthesize(self, job: SpeechJob) -> str | None:
        if job.provider == "edge":
            result = EdgeTTSProvider(self.config).synthesize_to_audio(
                text=job.text,
                voice=self._edge_voice_for_language(job.language),
            )

            if result.success:
                return result.audio_path

            return None

        if job.provider == "piper":
            provider = PiperTTSProvider(self.config)
            result = provider.synthesize_to_wav(
                job.text,
                model_path=job.model_path,
            )

            if result.success:
                return result.wav_path

            return None

        return None

    def _play_loop(self) -> None:
        player = AudioPlayer(self.config.tts_output_player)

        while not self._stop_event.is_set():
            job = self._play_queue.get()

            if job is None:
                break

            try:
                player.play(job.audio_path)

            finally:
                try:
                    Path(job.audio_path).unlink(missing_ok=True)
                except OSError:
                    pass

    @staticmethod
    def _drain_queue(target_queue) -> None:
        while True:
            try:
                target_queue.get_nowait()
            except queue.Empty:
                break