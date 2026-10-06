from __future__ import annotations

import urllib.request
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PiperVoice:
    id: str
    title: str
    language: str
    quality: str
    model_path: str
    model_url: str
    config_url: str


PIPER_VOICE_CATALOG: list[PiperVoice] = [
    # Ukrainian
    PiperVoice(
        id="uk_tetiana_high",
        title="Ukrainian Tetiana",
        language="uk",
        quality="high",
        model_path="models/piper/uk_UA/tetiana/high/uk_UA-tetiana-high.onnx",
        model_url="https://huggingface.co/rhasspy/piper-voices/resolve/main/uk/uk_UA/tetiana/high/uk_UA-tetiana-high.onnx",
        config_url="https://huggingface.co/rhasspy/piper-voices/resolve/main/uk/uk_UA/tetiana/high/uk_UA-tetiana-high.onnx.json",
    ),
    PiperVoice(
        id="uk_mykyta_high",
        title="Ukrainian Mykyta",
        language="uk",
        quality="high",
        model_path="models/piper/uk_UA/mykyta/high/uk_UA-mykyta-high.onnx",
        model_url="https://huggingface.co/rhasspy/piper-voices/resolve/main/uk/uk_UA/mykyta/high/uk_UA-mykyta-high.onnx",
        config_url="https://huggingface.co/rhasspy/piper-voices/resolve/main/uk/uk_UA/mykyta/high/uk_UA-mykyta-high.onnx.json",
    ),
    PiperVoice(
        id="uk_oleksa_high",
        title="Ukrainian Oleksa",
        language="uk",
        quality="high",
        model_path="models/piper/uk_UA/oleksa/high/uk_UA-oleksa-high.onnx",
        model_url="https://huggingface.co/rhasspy/piper-voices/resolve/main/uk/uk_UA/oleksa/high/uk_UA-oleksa-high.onnx",
        config_url="https://huggingface.co/rhasspy/piper-voices/resolve/main/uk/uk_UA/oleksa/high/uk_UA-oleksa-high.onnx.json",
    ),
    PiperVoice(
        id="uk_ukrainian_tts_medium",
        title="Ukrainian TTS",
        language="uk",
        quality="medium",
        model_path="models/piper/uk_UA/ukrainian_tts/medium/uk_UA-ukrainian_tts-medium.onnx",
        model_url="https://huggingface.co/rhasspy/piper-voices/resolve/main/uk/uk_UA/ukrainian_tts/medium/uk_UA-ukrainian_tts-medium.onnx",
        config_url="https://huggingface.co/rhasspy/piper-voices/resolve/main/uk/uk_UA/ukrainian_tts/medium/uk_UA-ukrainian_tts-medium.onnx.json",
    ),

    # English
    PiperVoice(
        id="en_amy_medium",
        title="English Amy",
        language="en",
        quality="medium",
        model_path="models/piper/en_US/amy/medium/en_US-amy-medium.onnx",
        model_url="https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/amy/medium/en_US-amy-medium.onnx",
        config_url="https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/amy/medium/en_US-amy-medium.onnx.json",
    ),
    PiperVoice(
        id="en_lessac_medium",
        title="English Lessac",
        language="en",
        quality="medium",
        model_path="models/piper/en_US/lessac/medium/en_US-lessac-medium.onnx",
        model_url="https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx",
        config_url="https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json",
    ),

    # Russian
    PiperVoice(
        id="ru_irina_medium",
        title="Russian Irina",
        language="ru",
        quality="medium",
        model_path="models/piper/ru_RU/irina/medium/ru_RU-irina-medium.onnx",
        model_url="https://huggingface.co/rhasspy/piper-voices/resolve/main/ru/ru_RU/irina/medium/ru_RU-irina-medium.onnx",
        config_url="https://huggingface.co/rhasspy/piper-voices/resolve/main/ru/ru_RU/irina/medium/ru_RU-irina-medium.onnx.json",
    ),
    PiperVoice(
        id="ru_dmitri_medium",
        title="Russian Dmitri",
        language="ru",
        quality="medium",
        model_path="models/piper/ru_RU/dmitri/medium/ru_RU-dmitri-medium.onnx",
        model_url="https://huggingface.co/rhasspy/piper-voices/resolve/main/ru/ru_RU/dmitri/medium/ru_RU-dmitri-medium.onnx",
        config_url="https://huggingface.co/rhasspy/piper-voices/resolve/main/ru/ru_RU/dmitri/medium/ru_RU-dmitri-medium.onnx.json",
    ),
    PiperVoice(
        id="ru_denis_medium",
        title="Russian Denis",
        language="ru",
        quality="medium",
        model_path="models/piper/ru_RU/denis/medium/ru_RU-denis-medium.onnx",
        model_url="https://huggingface.co/rhasspy/piper-voices/resolve/main/ru/ru_RU/denis/medium/ru_RU-denis-medium.onnx",
        config_url="https://huggingface.co/rhasspy/piper-voices/resolve/main/ru/ru_RU/denis/medium/ru_RU-denis-medium.onnx.json",
    ),
]


def get_piper_voice(voice_id: str) -> PiperVoice | None:
    for voice in PIPER_VOICE_CATALOG:
        if voice.id == voice_id:
            return voice

    return None


def voices_for_language(language: str) -> list[PiperVoice]:
    return [
        voice
        for voice in PIPER_VOICE_CATALOG
        if voice.language == language
    ]


def download_piper_voice(voice: PiperVoice) -> tuple[bool, str]:
    model_path = Path(voice.model_path)
    config_path = Path(f"{voice.model_path}.json")

    try:
        model_path.parent.mkdir(parents=True, exist_ok=True)

        if not model_path.exists():
            urllib.request.urlretrieve(voice.model_url, model_path)

        if not config_path.exists():
            urllib.request.urlretrieve(voice.config_url, config_path)

        return True, f"Voice ready: {voice.title}"

    except Exception as error:
        return False, str(error)