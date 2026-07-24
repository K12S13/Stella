from __future__ import annotations

import urllib.request
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PiperVoiceOption:
    id: str
    title: str
    model_path: str
    model_url: str
    config_url: str
    quality: str
    notes: str


PIPER_VOICE_CATALOG: list[PiperVoiceOption] = [
    PiperVoiceOption(
        id="ukrainian_tts_medium",
        title="Ukrainian TTS medium",
        model_path="models/piper/uk_UA/ukrainian_tts/medium/uk_UA-ukrainian_tts-medium.onnx",
        model_url="https://huggingface.co/rhasspy/piper-voices/resolve/main/uk/uk_UA/ukrainian_tts/medium/uk_UA-ukrainian_tts-medium.onnx",
        config_url="https://huggingface.co/rhasspy/piper-voices/resolve/main/uk/uk_UA/ukrainian_tts/medium/uk_UA-ukrainian_tts-medium.onnx.json",
        quality="medium",
        notes="Той голос, який ти вже тестував.",
    ),
    PiperVoiceOption(
        id="lada_x_low",
        title="Lada x-low",
        model_path="models/piper/uk_UA/lada/x_low/uk_UA-lada-x_low.onnx",
        model_url="https://huggingface.co/rhasspy/piper-voices/resolve/main/uk/uk_UA/lada/x_low/uk_UA-lada-x_low.onnx",
        config_url="https://huggingface.co/rhasspy/piper-voices/resolve/main/uk/uk_UA/lada/x_low/uk_UA-lada-x_low.onnx.json",
        quality="x-low",
        notes="Легкий голос, має бути швидкий.",
    ),
    PiperVoiceOption(
        id="mykyta_high",
        title="Mykyta high",
        model_path="models/piper/uk_UA/mykyta/high/uk_UA-mykyta-high.onnx",
        model_url="https://huggingface.co/rhasspy/piper-voices/resolve/main/uk/uk_UA/mykyta/high/uk_UA-mykyta-high.onnx",
        config_url="https://huggingface.co/rhasspy/piper-voices/resolve/main/uk/uk_UA/mykyta/high/uk_UA-mykyta-high.onnx.json",
        quality="high",
        notes="Український чоловічий high voice.",
    ),
    PiperVoiceOption(
        id="oleksa_high",
        title="Oleksa high",
        model_path="models/piper/uk_UA/oleksa/high/uk_UA-oleksa-high.onnx",
        model_url="https://huggingface.co/rhasspy/piper-voices/resolve/main/uk/uk_UA/oleksa/high/uk_UA-oleksa-high.onnx",
        config_url="https://huggingface.co/rhasspy/piper-voices/resolve/main/uk/uk_UA/oleksa/high/uk_UA-oleksa-high.onnx.json",
        quality="high",
        notes="Український high voice.",
    ),
    PiperVoiceOption(
        id="tetiana_high",
        title="Tetiana high",
        model_path="models/piper/uk_UA/tetiana/high/uk_UA-tetiana-high.onnx",
        model_url="https://huggingface.co/rhasspy/piper-voices/resolve/main/uk/uk_UA/tetiana/high/uk_UA-tetiana-high.onnx",
        config_url="https://huggingface.co/rhasspy/piper-voices/resolve/main/uk/uk_UA/tetiana/high/uk_UA-tetiana-high.onnx.json",
        quality="high",
        notes="Український жіночий high voice.",
    ),
]


def get_piper_voice(voice_id: str) -> PiperVoiceOption | None:
    for voice in PIPER_VOICE_CATALOG:
        if voice.id == voice_id:
            return voice

    return None


def _download_file(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)

    with urllib.request.urlopen(url, timeout=120) as response:
        with destination.open("wb") as file:
            while True:
                chunk = response.read(1024 * 1024)

                if not chunk:
                    break

                file.write(chunk)


def download_piper_voice(voice: PiperVoiceOption) -> tuple[bool, str]:
    model_path = Path(voice.model_path)
    config_path = Path(f"{voice.model_path}.json")

    try:
        if not model_path.exists():
            _download_file(voice.model_url, model_path)

        if not config_path.exists():
            _download_file(voice.config_url, config_path)

        return True, voice.model_path

    except Exception as error:
        return False, str(error)