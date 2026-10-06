from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class StellaConfig:
    assistant_name: str
    language: str
    resource_mode: str
    cloud_mode: str
    confirm_dangerous_actions: bool
    allow_shell_commands: bool
    local_provider: str
    local_model: str
    cloud_provider: str
    ollama_base_url: str
    llm_timeout_seconds: int
    ollama_keep_alive: str
    ollama_num_ctx: int
    stt_provider: str
    stt_mode: str
    stt_auto_submit: bool
    stt_listen_seconds: int
    stt_sample_rate: int
    stt_device_index: int | None
    stt_vosk_model_path: str
    stt_whisper_model: str
    stt_whisper_device: str
    stt_whisper_compute_type: str
    stt_whisper_language: str
    stt_whisper_beam_size: int
    stt_whisper_vad_filter: bool
    wake_enabled: bool
    wake_provider: str
    wake_words: list[str]
    wake_silence_seconds: float
    wake_min_command_seconds: float
    wake_max_command_seconds: int
    wake_energy_threshold: int
    wake_preroll_seconds: float
    wake_strict_confirm: bool
    wake_allow_fuzzy: bool
    wake_trigger_max_words_before_wake: int
    tts_provider: str
    tts_enabled: bool
    tts_speak_llm_answers: bool
    tts_speak_command_results: bool
    tts_piper_model: str
    tts_output_player: str
    tts_language_routing: bool
    tts_default_language: str
    tts_edge_uk_voice: str
    tts_edge_en_voice: str
    tts_edge_ru_voice: str
    tts_edge_rate: str
    tts_edge_volume: str
    tts_edge_pitch: str
    tts_fallback_provider: str

    tts_uk_provider: str
    tts_uk_model: str

    tts_en_provider: str
    tts_en_model: str

    tts_ru_provider: str
    tts_ru_model: str


def _get(data: dict[str, Any], path: str, default: Any = None) -> Any:
    current: Any = data

    for part in path.split("."):
        if not isinstance(current, dict):
            return default

        current = current.get(part)

        if current is None:
            return default

    return current

def _optional_int(value) -> int | None:
    if value is None:
        return None

    if isinstance(value, str) and value.strip().lower() in {"", "none", "null"}:
        return None

    return int(value)


def load_config(path: str | Path = "stella/data/settings.yaml") -> StellaConfig:
    config_path = Path(path)

    if not config_path.exists():
        raise FileNotFoundError(f"Settings file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}

    return StellaConfig(
        assistant_name=str(_get(data, "assistant.name", "стелла")).lower().strip(),
        language=str(_get(data, "assistant.language", "uk")).lower().strip(),
        resource_mode=str(_get(data, "modes.resource_mode", "balanced")).lower().strip(),
        cloud_mode=str(_get(data, "modes.cloud_mode", "fallback")).lower().strip(),
        confirm_dangerous_actions=bool(_get(data, "safety.confirm_dangerous_actions", True)),
        allow_shell_commands=bool(_get(data, "safety.allow_shell_commands", True)),
        local_provider=str(_get(data, "llm.local_provider", "ollama")).lower().strip(),
        local_model=str(_get(data, "llm.local_model", "qwen2.5:3b")).strip(),
        cloud_provider=str(_get(data, "llm.cloud_provider", "none")).lower().strip(),
        ollama_base_url=str(_get(data, "llm.ollama_base_url", "http://127.0.0.1:11434")).rstrip("/"),
        llm_timeout_seconds=int(_get(data, "llm.timeout_seconds", 45)),
        ollama_keep_alive=str(_get(data, "llm.keep_alive", "0")),
        ollama_num_ctx=int(_get(data, "llm.num_ctx", 2048)),
        stt_provider=str(_get(data, "stt.provider", "none")).lower().strip(),
        stt_mode=str(_get(data, "stt.mode", "push_to_talk")).lower().strip(),
        stt_auto_submit=bool(_get(data, "stt.auto_submit", False)),
        stt_listen_seconds=int(_get(data, "stt.listen_seconds", 5)),
        stt_sample_rate=int(_get(data, "stt.sample_rate", 16000)),
        stt_device_index=_optional_int(_get(data, "stt.device_index", None)),
        stt_vosk_model_path=str(_get(data, "stt.vosk_model_path", "models/vosk/uk-small")),
        stt_whisper_model=str(_get(data, "stt.whisper_model", "small")),
        stt_whisper_device=str(_get(data, "stt.whisper_device", "auto")),
        stt_whisper_compute_type=str(_get(data, "stt.whisper_compute_type", "auto")),
        stt_whisper_language=str(_get(data, "stt.whisper_language", "uk")),
        stt_whisper_beam_size=int(_get(data, "stt.whisper_beam_size", 5)),
        stt_whisper_vad_filter=bool(_get(data, "stt.whisper_vad_filter", True)),
        wake_enabled=bool(_get(data, "wake.enabled", False)),
        wake_provider=str(_get(data, "wake.provider", "vosk")),
        wake_words=list(_get(data, "wake.words", ["стелла", "стела", "stella"])),
        wake_silence_seconds=float(_get(data, "wake.silence_seconds", 1.1)),
        wake_min_command_seconds=float(_get(data, "wake.min_command_seconds", 1.0)),
        wake_max_command_seconds=int(_get(data, "wake.max_command_seconds", 10)),
        wake_energy_threshold=int(_get(data, "wake.energy_threshold", 350)),
        wake_preroll_seconds=float(_get(data, "wake.preroll_seconds", 1.0)),
        wake_strict_confirm=bool(_get(data, "wake.strict_confirm", True)),
        wake_allow_fuzzy=bool(_get(data, "wake.allow_fuzzy", False)),
        wake_trigger_max_words_before_wake=int(_get(data, "wake.trigger_max_words_before_wake", 1)),
        tts_provider=str(_get(data, "tts.provider", "none")).lower().strip(),
        tts_enabled=bool(_get(data, "tts.enabled", False)),
        tts_speak_llm_answers=bool(_get(data, "tts.speak_llm_answers", True)),
        tts_speak_command_results=bool(_get(data, "tts.speak_command_results", False)),
        tts_piper_model=str(_get(data, "tts.piper_model", "")).strip(),
        tts_output_player=str(_get(data, "tts.output_player", "pw-play")).strip(),
        tts_language_routing=bool(_get(data, "tts.language_routing", True)),
        tts_default_language=str(_get(data, "tts.default_language", "uk")),
        tts_edge_uk_voice=str(_get(data, "tts.edge.uk_voice", "uk-UA-PolinaNeural")),
        tts_edge_en_voice=str(_get(data, "tts.edge.en_voice", "en-US-JennyNeural")),
        tts_edge_ru_voice=str(_get(data, "tts.edge.ru_voice", "ru-RU-SvetlanaNeural")),
        tts_edge_rate=str(_get(data, "tts.edge.rate", "+0%")),
        tts_edge_volume=str(_get(data, "tts.edge.volume", "+0%")),
        tts_edge_pitch=str(_get(data, "tts.edge.pitch", "+0Hz")),
        tts_fallback_provider=str(_get(data, "tts.fallback_provider", "piper")),

        tts_uk_provider=str(_get(data, "tts.uk.provider", "piper")),
        tts_uk_model=str(
            _get(
                data,
                "tts.uk.model",
                "models/piper/uk_UA/tetiana/high/uk_UA-tetiana-high.onnx",
            )
        ),

        tts_en_provider=str(_get(data, "tts.en.provider", "piper")),
        tts_en_model=str(
            _get(
                data,
                "tts.en.model",
                "models/piper/en_US/amy/medium/en_US-amy-medium.onnx",
            )
        ),

        tts_ru_provider=str(_get(data, "tts.ru.provider", "piper")),
        tts_ru_model=str(
            _get(
                data,
                "tts.ru.model",
                "models/piper/ru_RU/irina/medium/ru_RU-irina-medium.onnx",
            )
        ),

    )