from __future__ import annotations

import re
from pathlib import Path

import yaml


PROFILE_PATH = Path("stella/data/wake_profile.yaml")


def normalize_wake_text(text: str) -> str:
    lowered = text.lower().strip()
    lowered = lowered.replace("ё", "е")
    lowered = re.sub(r"[^a-zа-яіїєґ\s]", " ", lowered, flags=re.IGNORECASE)
    lowered = re.sub(r"\s+", " ", lowered)
    return lowered.strip()


def load_wake_profile() -> dict:
    if not PROFILE_PATH.exists():
        return {
            "assistant_name": "",
            "aliases": [],
            "verification_aliases": [],
            "blocked_aliases": [],
            "accept_alias_without_whisper_wake": True,
        }

    data = yaml.safe_load(PROFILE_PATH.read_text(encoding="utf-8")) or {}

    return {
        "assistant_name": str(data.get("assistant_name", "")),
        "aliases": list(data.get("aliases", [])),
        "verification_aliases": list(data.get("verification_aliases", [])),
        "blocked_aliases": list(data.get("blocked_aliases", [])),
        "accept_alias_without_whisper_wake": bool(
            data.get("accept_alias_without_whisper_wake", True)
        ),
    }


def save_wake_profile(profile: dict) -> None:
    PROFILE_PATH.parent.mkdir(parents=True, exist_ok=True)

    normalized_profile = {
        "assistant_name": normalize_wake_text(str(profile.get("assistant_name", ""))),
        "aliases": _unique_normalized(profile.get("aliases", [])),
        "verification_aliases": _unique_normalized(
            profile.get("verification_aliases", [])
        ),
        "blocked_aliases": _unique_normalized(profile.get("blocked_aliases", [])),
        "accept_alias_without_whisper_wake": bool(
            profile.get("accept_alias_without_whisper_wake", True)
        ),
    }

    PROFILE_PATH.write_text(
        yaml.safe_dump(
            normalized_profile,
            allow_unicode=True,
            sort_keys=False,
        ),
        encoding="utf-8",
    )


def _unique_normalized(items: list[str]) -> list[str]:
    result: list[str] = []

    for item in items:
        normalized = normalize_wake_text(str(item))

        if normalized and normalized not in result:
            result.append(normalized)

    return result