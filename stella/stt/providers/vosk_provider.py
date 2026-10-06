from __future__ import annotations

import json
import queue
import threading
import time
from pathlib import Path

import sounddevice as sd
from vosk import KaldiRecognizer, Model, SetLogLevel

from stella.stt.providers.base import BaseSTTProvider, STTResult


SetLogLevel(-1)

_MODEL_CACHE: dict[str, Model] = {}


def list_input_devices() -> list[str]:
    devices = sd.query_devices()
    result: list[str] = []

    for index, device in enumerate(devices):
        max_input_channels = int(device.get("max_input_channels", 0))

        if max_input_channels > 0:
            name = str(device.get("name", "Unknown input device"))
            default_samplerate = int(float(device.get("default_samplerate", 0)))
            result.append(f"{index}: {name} | {default_samplerate} Hz")

    return result


class VoskProvider(BaseSTTProvider):
    name = "vosk"

    def __init__(
        self,
        model_path: str,
        sample_rate: int = 16000,
        device_index: int | None = None,
    ) -> None:
        self.model_path = Path(model_path)
        self.sample_rate = sample_rate
        self.device_index = device_index

    def _load_model(self) -> Model:
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Vosk model not found: {self.model_path}. "
                "Expected path example: models/vosk/uk-small"
            )

        key = str(self.model_path.resolve())

        if key not in _MODEL_CACHE:
            _MODEL_CACHE[key] = Model(key)

        return _MODEL_CACHE[key]

    def listen_once(
        self,
        seconds: int = 30,
        stop_event: threading.Event | None = None,
    ) -> STTResult:
        audio_queue: queue.Queue[bytes] = queue.Queue()

        try:
            model = self._load_model()
            recognizer = KaldiRecognizer(model, self.sample_rate)

            def callback(indata, frames, callback_time, status) -> None:
                audio_queue.put(bytes(indata))

            recognized_parts: list[str] = []

            max_seconds = max(1, int(seconds))
            deadline = time.monotonic() + max_seconds

            with sd.RawInputStream(
                samplerate=self.sample_rate,
                blocksize=8000,
                dtype="int16",
                channels=1,
                device=self.device_index,
                callback=callback,
            ):
                while time.monotonic() < deadline:
                    if stop_event is not None and stop_event.is_set():
                        break

                    try:
                        data = audio_queue.get(timeout=0.2)
                    except queue.Empty:
                        continue

                    if recognizer.AcceptWaveform(data):
                        chunk_result = json.loads(recognizer.Result())
                        chunk_text = str(chunk_result.get("text", "")).strip()

                        if chunk_text:
                            recognized_parts.append(chunk_text)

                final_result = json.loads(recognizer.FinalResult())
                final_text = str(final_result.get("text", "")).strip()

                if final_text:
                    recognized_parts.append(final_text)

            text = " ".join(part for part in recognized_parts if part).strip()

            if not text:
                return STTResult(
                    success=False,
                    text="",
                    error=(
                        "Vosk нічого не розпізнав. Спробуй говорити голосніше, "
                        "збільшити час запису або вибрати інший STT provider."
                    ),
                    provider=self.name,
                )

            return STTResult(
                success=True,
                text=text,
                provider=self.name,
            )

        except Exception as error:
            return STTResult(
                success=False,
                text="",
                error=str(error),
                provider=self.name,
            )