from __future__ import annotations

import json
import queue
import re
import tempfile
import threading
import time
import wave
from collections import deque
from pathlib import Path

import numpy as np
import sounddevice as sd
from PySide6.QtCore import QObject, Signal
from vosk import KaldiRecognizer, Model, SetLogLevel

from stella.stt.wake_profile import load_wake_profile, normalize_wake_text
from stella.core.config import StellaConfig
from stella.stt.whisper_file_transcriber import transcribe_wav_with_whisper
from stella.text.stt_text_cleanup import cleanup_stt_text


_VOSK_MODEL_CACHE: dict[str, Model] = {}


def _load_vosk_model(model_path: str) -> Model:
    if model_path in _VOSK_MODEL_CACHE:
        return _VOSK_MODEL_CACHE[model_path]

    path = Path(model_path)

    if not path.exists():
        raise RuntimeError(f"Vosk wake model not found: {model_path}")

    model = Model(str(path))
    _VOSK_MODEL_CACHE[model_path] = model
    return model


class WakeCommandWorker(QObject):
    status = Signal(str)
    recognized = Signal(str)
    failed = Signal(str)
    finished = Signal()

    def __init__(self, config: StellaConfig) -> None:
        super().__init__()
        self.config = config
        self.stop_event = threading.Event()

        self.last_debug_at = 0.0
        self.last_partial_text = ""

        self.last_wake_trigger_text = ""
        self.last_wake_trigger_strength = ""

        self.cooldown_until = 0.0
        self.wake_profile = load_wake_profile()

    def stop(self) -> None:
        self.stop_event.set()

    def run(self) -> None:
        try:
            SetLogLevel(-1)

            if self.config.wake_provider != "vosk":
                self.failed.emit(f"Wake provider not supported: {self.config.wake_provider}")
                return

            model = _load_vosk_model(self.config.stt_vosk_model_path)

            grammar = [
                "стелла",
                "стела",
                "stella",
                "[unk]",
            ]

            recognizer = KaldiRecognizer(
                model,
                self.config.stt_sample_rate,
                json.dumps(grammar, ensure_ascii=False),
            )

            audio_queue: queue.Queue[bytes] = queue.Queue()

            def callback(indata, frames, callback_time, status) -> None:
                audio_queue.put(bytes(indata))

            blocksize = 4000
            chunks_per_second = max(1, int(self.config.stt_sample_rate / blocksize))

            preroll_chunks = max(
                1,
                int(float(getattr(self.config, "wake_preroll_seconds", 4.0)) * chunks_per_second),
            )

            rolling_audio: deque[bytes] = deque(maxlen=preroll_chunks)

            # ~1 second energy window.
            energy_window: deque[float] = deque(maxlen=chunks_per_second)

            self.status.emit("Wake listener active. Say: Стелла + command")

            with sd.RawInputStream(
                samplerate=self.config.stt_sample_rate,
                blocksize=blocksize,
                dtype="int16",
                channels=1,
                device=self.config.stt_device_index,
                callback=callback,
            ):
                while not self.stop_event.is_set():
                    try:
                        audio = audio_queue.get(timeout=0.2)
                    except queue.Empty:
                        continue

                    rolling_audio.append(audio)

                    energy = self._audio_energy(audio)
                    energy_window.append(energy)

                    max_recent_energy = max(energy_window) if energy_window else energy
                    self._debug_energy(energy)

                    if recognizer.AcceptWaveform(audio):
                        payload = json.loads(recognizer.Result())
                        text = payload.get("text", "")
                    else:
                        payload = json.loads(recognizer.PartialResult())
                        text = payload.get("partial", "")

                    text = text.strip()

                    if text and text != self.last_partial_text:
                        self.last_partial_text = text
                        self.status.emit(f"Wake heard: {text}")

                    if time.monotonic() < self.cooldown_until:
                        continue

                    trigger_strength = self._wake_trigger_strength(
                        text=text,
                        max_recent_energy=max_recent_energy,
                    )

                    if trigger_strength is None:
                        continue

                    self.last_wake_trigger_text = text
                    self.last_wake_trigger_strength = trigger_strength

                    self.status.emit(
                        f"Wake word detected [{trigger_strength}]: {text}. Capturing full command..."
                    )

                    command_frames = list(rolling_audio)

                    wav_path = self._record_command_tail(
                        audio_queue=audio_queue,
                        command_frames=command_frames,
                    )

                    if wav_path is None:
                        self._reject_and_continue(
                            recognizer=recognizer,
                            rolling_audio=rolling_audio,
                            audio_queue=audio_queue,
                            reason="Wake detected, but command recording failed.",
                        )
                        continue

                    try:
                        result = transcribe_wav_with_whisper(self.config, wav_path)

                        if not result.success:
                            self._reject_and_continue(
                                recognizer=recognizer,
                                rolling_audio=rolling_audio,
                                audio_queue=audio_queue,
                                reason=result.error or "Whisper command recognition failed.",
                            )
                            continue

                        cleaned = cleanup_stt_text(result.text)
                        self.status.emit(f"Whisper wake check heard: {cleaned}")

                        if getattr(self.config, "wake_strict_confirm", True):
                            command = self._extract_command_after_verified_wake(cleaned)

                            if command is None:
                                if getattr(self.config, "wake_strict_confirm", True):
                                    command = self._extract_command_after_verified_wake(cleaned)

                                    if command is None:
                                        accept_calibrated = bool(
                                            self.wake_profile.get(
                                                "accept_alias_without_whisper_wake",
                                                False,
                                            )
                                        )

                                        if (
                                                accept_calibrated
                                                and self.last_wake_trigger_strength in {"exact", "fuzzy"}
                                                and self.last_wake_trigger_text
                                        ):
                                            self.status.emit(
                                                "Whisper missed wake word, but calibrated wake profile accepted trigger. "
                                                f"Using recognized text as command: {cleaned}"
                                            )
                                            command = cleaned
                                        else:
                                            self._reject_and_continue(
                                                recognizer=recognizer,
                                                rolling_audio=rolling_audio,
                                                audio_queue=audio_queue,
                                                reason=(
                                                    f"False wake rejected by Whisper "
                                                    f"[vosk={self.last_wake_trigger_text}]: {cleaned}"
                                                ),
                                            )
                                            continue

                                    cleaned = command
                                else:
                                    cleaned = self._remove_wake_phrase(cleaned)

                        cleaned = cleanup_stt_text(cleaned)

                        if not cleaned:
                            self._reject_and_continue(
                                recognizer=recognizer,
                                rolling_audio=rolling_audio,
                                audio_queue=audio_queue,
                                reason="Wake detected, but no command after wake word.",
                            )
                            continue

                        self.recognized.emit(cleaned)
                        return

                    finally:
                        try:
                            wav_path.unlink(missing_ok=True)
                        except OSError:
                            pass

        except Exception as error:
            self.failed.emit(str(error))

        finally:
            self.finished.emit()

    def _wake_trigger_strength(
        self,
        text: str,
        max_recent_energy: float,
    ) -> str | None:
        normalized = self._normalize_text(text)

        if not normalized:
            return None

        min_trigger_energy = float(getattr(self.config, "wake_min_trigger_energy", 180))

        if max_recent_energy < min_trigger_energy:
            return None

        compact = normalized.replace(" ", "")
        wake_variants = self._wake_variants()

        for wake in wake_variants:
            wake_norm = self._normalize_text(wake)

            if not wake_norm:
                continue

            wake_compact = wake_norm.replace(" ", "")

            # Exact token.
            if re.search(rf"(^|\s){re.escape(wake_norm)}($|\s)", normalized):
                return "exact"

            # Exact compact: "стелла" / "stella".
            if wake_compact and wake_compact == compact:
                return "exact"

        if not getattr(self.config, "wake_allow_fuzzy", False):
            return None

        tokens = normalized.split()

        for token in tokens:
            # Critical guard: "тела" must never trigger "стела".
            if not token.startswith("ст") and token != "stella":
                continue

            for wake in wake_variants:
                wake_token = self._normalize_text(wake).replace(" ", "")

                if not wake_token:
                    continue

                if abs(len(token) - len(wake_token)) > 1:
                    continue

                if self._levenshtein_distance(token, wake_token) <= 1:
                    return "fuzzy"

        return None

    def _wake_variants(self) -> set[str]:
        variants = {
            "стелла",
            "стела",
            "stella",
        }

        for word in self.config.wake_words:
            clean = normalize_wake_text(str(word))

            if clean:
                variants.add(clean)

        for word in self.wake_profile.get("aliases", []):
            clean = normalize_wake_text(str(word))

            if clean:
                variants.add(clean)

        for word in self.wake_profile.get("blocked_aliases", []):
            blocked = normalize_wake_text(str(word))

            if blocked in variants:
                variants.remove(blocked)

        return variants

    def _verification_wake_variants(self) -> set[str]:
        variants = {
            "стелла",
            "стела",
            "stella",
        }

        for word in self.config.wake_words:
            clean = normalize_wake_text(str(word))

            if clean:
                variants.add(clean)

        for word in self.wake_profile.get("verification_aliases", []):
            clean = normalize_wake_text(str(word))

            if clean:
                variants.add(clean)

        return variants

    def _extract_command_after_verified_wake(self, text: str) -> str | None:
        normalized = normalize_wake_text(text)

        if not normalized:
            return None

        wake_variants = sorted(
            self._verification_wake_variants(),
            key=len,
            reverse=True,
        )

        max_before = max(
            0,
            int(getattr(self.config, "wake_trigger_max_words_before_wake", 0)),
        )

        words = normalized.split()

        for start_index in range(0, min(len(words), max_before + 1)):
            tail = " ".join(words[start_index:])

            for wake in wake_variants:
                wake_norm = normalize_wake_text(wake)

                if not wake_norm:
                    continue

                if tail == wake_norm:
                    return ""

                if tail.startswith(wake_norm + " "):
                    return tail[len(wake_norm):].strip()

                tail_compact = tail.replace(" ", "")
                wake_compact = wake_norm.replace(" ", "")

                if wake_compact and tail_compact.startswith(wake_compact):
                    raw_command = tail_compact[len(wake_compact):].strip()

                    if len(raw_command) >= 3:
                        return raw_command

        return None

    def _remove_wake_phrase(self, text: str) -> str:
        result = text.strip()

        for wake in self._wake_variants():
            wake_norm = self._normalize_text(wake)

            if not wake_norm:
                continue

            result = re.sub(
                rf"^{re.escape(wake_norm)}[\s,.:;!?-]*",
                "",
                result,
                flags=re.IGNORECASE,
            ).strip()

        return result

    def _record_command_tail(
        self,
        audio_queue: queue.Queue[bytes],
        command_frames: list[bytes],
    ) -> Path | None:
        started = time.monotonic()
        last_voice_at = time.monotonic()

        max_seconds = max(1, int(self.config.wake_max_command_seconds))
        min_seconds = max(0.1, float(self.config.wake_min_command_seconds))
        silence_seconds = max(0.2, float(self.config.wake_silence_seconds))
        threshold = max(1, int(self.config.wake_energy_threshold))

        while not self.stop_event.is_set():
            elapsed = time.monotonic() - started

            if elapsed >= max_seconds:
                break

            try:
                audio = audio_queue.get(timeout=0.15)
            except queue.Empty:
                continue

            command_frames.append(audio)

            energy = self._audio_energy(audio)

            if energy >= threshold:
                last_voice_at = time.monotonic()

            enough_audio = elapsed >= min_seconds
            silence_long_enough = time.monotonic() - last_voice_at >= silence_seconds

            if enough_audio and silence_long_enough:
                break

        if not command_frames:
            return None

        temp_file = tempfile.NamedTemporaryFile(
            suffix=".wav",
            prefix="stella_wake_command_",
            delete=False,
        )
        wav_path = Path(temp_file.name)
        temp_file.close()

        with wave.open(str(wav_path), "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(self.config.stt_sample_rate)
            wav_file.writeframes(b"".join(command_frames))

        return wav_path

    def _reject_and_continue(
        self,
        recognizer: KaldiRecognizer,
        rolling_audio: deque[bytes],
        audio_queue: queue.Queue[bytes],
        reason: str,
    ) -> None:
        self.status.emit(reason)

        self.last_partial_text = ""
        self.last_wake_trigger_text = ""
        self.last_wake_trigger_strength = ""

        recognizer.Reset()
        rolling_audio.clear()
        self._drain_audio_queue(audio_queue)

        cooldown_seconds = float(getattr(self.config, "wake_cooldown_seconds", 1.5))
        self.cooldown_until = time.monotonic() + cooldown_seconds

    def _normalize_text(self, text: str) -> str:
        lowered = text.lower().strip()
        lowered = lowered.replace("ё", "е")
        lowered = re.sub(r"[^a-zа-яіїєґ\s]", " ", lowered, flags=re.IGNORECASE)
        lowered = re.sub(r"\s+", " ", lowered)
        return lowered.strip()

    def _audio_energy(self, audio: bytes) -> float:
        samples = np.frombuffer(audio, dtype=np.int16)

        if samples.size == 0:
            return 0.0

        return float(np.abs(samples).mean())

    def _debug_energy(self, energy: float) -> None:
        now = time.monotonic()

        if now - self.last_debug_at < 2.0:
            return

        self.last_debug_at = now
        self.status.emit(f"Wake mic energy: {energy:.1f}")

    def _drain_audio_queue(self, audio_queue: queue.Queue[bytes]) -> None:
        while True:
            try:
                audio_queue.get_nowait()
            except queue.Empty:
                break

    def _levenshtein_distance(self, left: str, right: str) -> int:
        if left == right:
            return 0

        if not left:
            return len(right)

        if not right:
            return len(left)

        previous = list(range(len(right) + 1))

        for i, left_char in enumerate(left, start=1):
            current = [i]

            for j, right_char in enumerate(right, start=1):
                insert_cost = current[j - 1] + 1
                delete_cost = previous[j] + 1
                replace_cost = previous[j - 1] + (left_char != right_char)

                current.append(min(insert_cost, delete_cost, replace_cost))

            previous = current

        return previous[-1]