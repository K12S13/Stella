from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass


@dataclass
class AudioPlayResult:
    success: bool
    message: str


class AudioPlayer:
    def __init__(self, configured_player: str = "") -> None:
        self.configured_player = configured_player.strip()

    def _player_command(self, audio_path: str) -> list[str] | None:
        if self.configured_player and shutil.which(self.configured_player):
            if self.configured_player == "mpv":
                return ["mpv", "--no-video", "--really-quiet", audio_path]

            return [self.configured_player, audio_path]

        if shutil.which("mpv"):
            return ["mpv", "--no-video", "--really-quiet", audio_path]

        for candidate in ["pw-play", "paplay", "aplay"]:
            if shutil.which(candidate):
                return [candidate, audio_path]

        return None

    def play(self, audio_path: str) -> AudioPlayResult:
        command = self._player_command(audio_path)

        if command is None:
            return AudioPlayResult(
                False,
                "No audio player found. Install mpv or use pw-play/paplay/aplay.",
            )

        result = subprocess.run(
            command,
            text=True,
            capture_output=True,
            check=False,
        )

        if result.returncode != 0:
            output = "\n".join(
                part.strip()
                for part in [result.stdout, result.stderr]
                if part.strip()
            )

            return AudioPlayResult(False, output or "Audio playback failed.")

        return AudioPlayResult(True, "Playback OK.")