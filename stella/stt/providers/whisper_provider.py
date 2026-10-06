from __future__ import annotations

import ctypes
import queue
import sys
import tempfile
import threading
import time
import wave
from pathlib import Path
from typing import Any

import sounddevice as sd

from stella.stt.providers.base import BaseSTTProvider, STTResult


_MODEL_CACHE: dict[tuple[str, str, str], Any] = {}


def _nvidia_library_dirs() -> list[Path]:
    dirs: list[Path] = []

    for base in [Path(sys.prefix), *[Path(p) for p in sys.path if p]]:
        if not base.exists():
            continue

        for path in base.rglob("site-packages/nvidia/*/lib"):
            if path.is_dir() and path not in dirs:
                dirs.append(path)

    return dirs


def _preload_cuda_libraries() -> None:
    dirs = _nvidia_library_dirs()

    if not dirs:
        raise RuntimeError(
            "NVIDIA pip libraries not found. Run: "
            "python -m pip install -U nvidia-cublas-cu12 nvidia-cudnn-cu12 nvidia-cuda-runtime-cu12"
        )

    required = [
        "libcudart.so.12",
        "libcublas.so.12",
        "libcublasLt.so.12",
        "libcudnn.so.9",
    ]

    optional = [
        "libcudnn_ops.so.9",
        "libcudnn_cnn.so.9",
        "libcudnn_adv.so.9",
        "libcudnn_graph.so.9",
        "libcudnn_engines_runtime_compiled.so.9",
        "libcudnn_engines_precompiled.so.9",
        "libcudnn_heuristic.so.9",
        "libcudnn_ext.so.9",
    ]

    missing: list[str] = []

    for lib_name in required:
        loaded = False

        for directory in dirs:
            candidate = directory / lib_name

            if not candidate.exists():
                continue

            ctypes.CDLL(str(candidate), mode=ctypes.RTLD_GLOBAL)
            loaded = True
            break

        if not loaded:
            missing.append(lib_name)

    if missing:
        raise RuntimeError(
            "CUDA libraries missing: "
            + ", ".join(missing)
            + ". Installed NVIDIA dirs: "
            + ", ".join(str(path) for path in dirs)
        )

    for lib_name in optional:
        for directory in dirs:
            candidate = directory / lib_name

            if candidate.exists():
                try:
                    ctypes.CDLL(str(candidate), mode=ctypes.RTLD_GLOBAL)
                except OSError:
                    pass

                break


class WhisperProvider(BaseSTTProvider):
    name = "whisper"

    def __init__(
        self,
        model_name: str = "large-v3-turbo",
        sample_rate: int = 16000,
        device_index: int | None = None,
        device: str = "cuda",
        compute_type: str = "int8_float16",
        language: str = "uk",
        beam_size: int = 1,
        vad_filter: bool = True,
    ) -> None:
        self.model_name = model_name
        self.sample_rate = sample_rate
        self.device_index = device_index
        self.device = device
        self.compute_type = compute_type
        self.language = language
        self.beam_size = beam_size
        self.vad_filter = vad_filter

    def _resolve_device(self) -> str:
        if self.device in {"cpu", "cuda"}:
            return self.device

        return "cuda"

    def _resolve_compute_type(self, device: str) -> str:
        if self.compute_type != "auto":
            return self.compute_type

        if device == "cuda":
            return "int8_float16"

        return "int8"

    def _load_model(self):
        device = self._resolve_device()
        compute_type = self._resolve_compute_type(device)
        key = (self.model_name, device, compute_type)

        if key in _MODEL_CACHE:
            return _MODEL_CACHE[key]

        if device == "cuda":
            _preload_cuda_libraries()

        from faster_whisper import WhisperModel

        try:
            model = WhisperModel(
                self.model_name,
                device=device,
                compute_type=compute_type,
            )

            _MODEL_CACHE[key] = model
            return model

        except Exception:
            if self.device == "cuda":
                raise

            fallback_key = (self.model_name, "cpu", "int8")

            if fallback_key not in _MODEL_CACHE:
                _MODEL_CACHE[fallback_key] = WhisperModel(
                    self.model_name,
                    device="cpu",
                    compute_type="int8",
                )

            return _MODEL_CACHE[fallback_key]

    def _record_wav(
        self,
        seconds: int,
        stop_event: threading.Event | None,
    ) -> Path:
        audio_queue: queue.Queue[bytes] = queue.Queue()

        def callback(indata, frames, callback_time, status) -> None:
            audio_queue.put(bytes(indata))

        temp_file = tempfile.NamedTemporaryFile(
            suffix=".wav",
            prefix="stella_whisper_",
            delete=False,
        )
        temp_path = Path(temp_file.name)
        temp_file.close()

        max_seconds = max(1, int(seconds))
        deadline = time.monotonic() + max_seconds
        frames: list[bytes] = []

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
                    frames.append(audio_queue.get(timeout=0.2))
                except queue.Empty:
                    continue

        with wave.open(str(temp_path), "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(self.sample_rate)
            wav_file.writeframes(b"".join(frames))

        return temp_path

    def listen_once(
        self,
        seconds: int = 60,
        stop_event: threading.Event | None = None,
    ) -> STTResult:
        wav_path: Path | None = None

        try:
            wav_path = self._record_wav(seconds=seconds, stop_event=stop_event)

            if wav_path.stat().st_size <= 44:
                return STTResult(
                    success=False,
                    text="",
                    error="Запис порожній. Мікрофон не дав аудіо.",
                    provider=self.name,
                )

            model = self._load_model()

            started = time.monotonic()

            segments, info = model.transcribe(
                str(wav_path),
                language=self.language or None,
                task="transcribe",
                beam_size=self.beam_size,
                vad_filter=self.vad_filter,
                condition_on_previous_text=False,
                temperature=0.0,
                initial_prompt=(
                    "Це українська голосова команда для desktop assistant Stella. "
                    "Розпізнай саме український текст. Не перекладай англійською."
                ),
            )

            text_parts: list[str] = []

            for segment in segments:
                part = segment.text.strip()

                if part:
                    text_parts.append(part)

            text = " ".join(text_parts).strip()
            elapsed = time.monotonic() - started

            if not text:
                return STTResult(
                    success=False,
                    text="",
                    error=f"Whisper нічого не розпізнав. Processing: {elapsed:.2f}s",
                    provider=self.name,
                )

            return STTResult(
                success=True,
                text=text,
                error=f"Processing: {elapsed:.2f}s",
                provider=self.name,
            )

        except Exception as error:
            return STTResult(
                success=False,
                text="",
                error=str(error),
                provider=self.name,
            )

        finally:
            if wav_path is not None:
                try:
                    wav_path.unlink(missing_ok=True)
                except OSError:
                    pass

def preload_whisper_model(config) -> tuple[bool, str]:
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

            provider._load_model()

            return True, (
                f"Whisper preloaded: "
                f"{config.stt_whisper_model} / "
                f"{config.stt_whisper_device} / "
                f"{config.stt_whisper_compute_type}"
            )

        except Exception as error:
            return False, f"Whisper preload failed: {error}"