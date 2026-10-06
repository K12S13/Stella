from __future__ import annotations

import json
import queue
import tempfile
import time
import wave
from collections import Counter
from pathlib import Path

import numpy as np
import sounddevice as sd
from PySide6.QtCore import QObject, Signal
from vosk import KaldiRecognizer, Model, SetLogLevel

from stella.core.config import StellaConfig
from stella.stt.wake_profile import (
    load_wake_profile,
    normalize_wake_text,
    save_wake_profile,
)


class WakeCalibrationWorker(QObject):
    status = Signal(str)
    sample_result = Signal(int, str, str, float, int)
    finished = Signal(bool, str)

    def __init__(
        self,
        config: StellaConfig,
        assistant_name: str,
        samples_count: int = 8,
        seconds_per_sample: float = 1.2,
    ) -> None:
        super().__init__()
        self.config = config
        self.assistant_name = normalize_wake_text(assistant_name)
        self.samples_count = samples_count
        self.seconds_per_sample = seconds_per_sample
        self.cancel_requested = False

    def cancel(self) -> None:
        self.cancel_requested = True

    def run(self) -> None:
        try:
            SetLogLevel(-1)

            if not self.assistant_name:
                self.finished.emit(False, "Assistant name is empty.")
                return

            model_path = Path(self.config.stt_vosk_model_path)

            if not model_path.exists():
                self.finished.emit(False, f"Vosk model not found: {model_path}")
                return

            self.status.emit("Loading Vosk wake model...")
            vosk_model = Model(str(model_path))

            vosk_results: list[str] = []

            for index in range(1, self.samples_count + 1):
                if self.cancel_requested:
                    self.finished.emit(False, "Wake calibration cancelled.")
                    return

                self.status.emit(f"Sample {index}/{self.samples_count}: prepare...")
                time.sleep(0.5)

                self.status.emit(
                    f"Sample {index}/{self.samples_count}: SAY '{self.assistant_name}' NOW"
                )

                wav_path, mean_energy, peak = self._record_wav_safe()

                try:
                    vosk_text = self._vosk_transcribe(
                        wav_path=wav_path,
                        vosk_model=vosk_model,
                    )

                    if vosk_text:
                        vosk_results.append(vosk_text)

                    self.sample_result.emit(
                        index,
                        vosk_text or "<empty>",
                        "<skipped>",
                        mean_energy,
                        peak,
                    )

                finally:
                    wav_path.unlink(missing_ok=True)

            message = self._save_profile(vosk_results)
            self.finished.emit(True, message)

        except Exception as error:
            self.finished.emit(False, str(error))

    def _record_wav_safe(self) -> tuple[Path, float, int]:
        audio_queue: queue.Queue[bytes] = queue.Queue()

        def callback(indata, frames, callback_time, status) -> None:
            audio_queue.put(bytes(indata))

        frames_data: list[bytes] = []
        started = time.monotonic()
        deadline = started + self.seconds_per_sample

        with sd.RawInputStream(
            samplerate=self.config.stt_sample_rate,
            blocksize=4000,
            dtype="int16",
            channels=1,
            device=self.config.stt_device_index,
            callback=callback,
        ):
            while time.monotonic() < deadline:
                if self.cancel_requested:
                    break

                try:
                    frames_data.append(audio_queue.get(timeout=0.2))
                except queue.Empty:
                    continue

        audio_bytes = b"".join(frames_data)

        if audio_bytes:
            samples = np.frombuffer(audio_bytes, dtype=np.int16)
            mean_energy = float(np.abs(samples).mean())
            peak = int(np.abs(samples).max())
        else:
            mean_energy = 0.0
            peak = 0

        temp_file = tempfile.NamedTemporaryFile(
            suffix=".wav",
            prefix="stella_wake_calibration_",
            delete=False,
        )
        wav_path = Path(temp_file.name)
        temp_file.close()

        with wave.open(str(wav_path), "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(self.config.stt_sample_rate)
            wav_file.writeframes(audio_bytes)

        return wav_path, mean_energy, peak

    def _vosk_transcribe(self, wav_path: Path, vosk_model: Model) -> str:
        grammar = [
            self.assistant_name,
            "стелла",
            "стела",
            "stella",
            "[unk]",
        ]

        recognizer = KaldiRecognizer(
            vosk_model,
            self.config.stt_sample_rate,
            json.dumps(grammar, ensure_ascii=False),
        )

        with wave.open(str(wav_path), "rb") as wav_file:
            data = wav_file.readframes(wav_file.getnframes())

        recognizer.AcceptWaveform(data)
        payload = json.loads(recognizer.FinalResult())

        return normalize_wake_text(payload.get("text", ""))

    def _save_profile(self, vosk_results: list[str]) -> str:
        vosk_counter = Counter(vosk_results)

        base_aliases = [
            self.assistant_name,
            "стелла",
            "стела",
            "stella",
        ]

        calibrated_aliases = [
            text
            for text, count in vosk_counter.most_common()
            if text and text != "<empty>" and count >= 2
        ]

        if not calibrated_aliases and vosk_counter:
            calibrated_aliases.append(vosk_counter.most_common(1)[0][0])

        aliases: list[str] = []

        for item in [*base_aliases, *calibrated_aliases]:
            normalized = normalize_wake_text(item)

            if normalized and normalized not in aliases:
                aliases.append(normalized)

        old_profile = load_wake_profile()

        profile = {
            "assistant_name": self.assistant_name,
            "aliases": aliases,
            "verification_aliases": [
                self.assistant_name,
                "стелла",
                "стела",
                "stella",
            ],
            "blocked_aliases": old_profile.get("blocked_aliases", []),
            "accept_alias_without_whisper_wake": True,
        }

        save_wake_profile(profile)

        stats = ", ".join(
            f"{text}={count}"
            for text, count in vosk_counter.most_common()
        )

        return (
            "Wake calibration saved.\n"
            f"Aliases: {', '.join(aliases) or 'none'}\n"
            f"Vosk stats: {stats or 'empty'}"
        )