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
    tts_provider: str
    tts_enabled: bool
    tts_speak_llm_answers: bool
    tts_speak_command_results: bool
    tts_piper_model: str
    tts_output_player: str


def _get(data: dict[str, Any], path: str, default: Any = None) -> Any:
    current: Any = data

    for part in path.split("."):
        if not isinstance(current, dict):
            return default

        current = current.get(part)

        if current is None:
            return default

    return current


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
        tts_provider=str(_get(data, "tts.provider", "none")).lower().strip(),
        tts_enabled=bool(_get(data, "tts.enabled", False)),
        tts_speak_llm_answers=bool(_get(data, "tts.speak_llm_answers", True)),
        tts_speak_command_results=bool(_get(data, "tts.speak_command_results", False)),
        tts_piper_model=str(_get(data, "tts.piper_model", "")).strip(),
        tts_output_player=str(_get(data, "tts.output_player", "pw-play")).strip(),
    )