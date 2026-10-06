from __future__ import annotations

import asyncio
import tempfile
from dataclasses import dataclass
from pathlib import Path

import edge_tts

from stella.core.config import StellaConfig
from stella.tts.audio_player import AudioPlayer


@dataclass
class EdgeTTSResult:
    success: bool
    message: str
    audio_path: str | None = None


class EdgeTTSProvider:
    def __init__(self, config: StellaConfig) -> None:
        self.config = config

    def synthesize_to_audio(
        self,
        text: str,
        voice: str,
    ) -> EdgeTTSResult:
        clean_text = text.strip()

        if not clean_text:
            return EdgeTTSResult(False, "Edge TTS text is empty.")

        if not voice.strip():
            return EdgeTTSResult(False, "Edge TTS voice is empty.")

        with tempfile.NamedTemporaryFile(
            suffix=".mp3",
            prefix="stella_edge_tts_",
            delete=False,
        ) as temp_file:
            audio_path = Path(temp_file.name)

        try:
            asyncio.run(
                self._save_audio(
                    text=clean_text,
                    voice=voice.strip(),
                    audio_path=audio_path,
                )
            )

            if not audio_path.exists() or audio_path.stat().st_size == 0:
                return EdgeTTSResult(False, "Edge TTS generated empty audio.")

            return EdgeTTSResult(True, "Edge TTS OK.", str(audio_path))

        except Exception as error:
            try:
                audio_path.unlink(missing_ok=True)
            except OSError:
                pass

            return EdgeTTSResult(False, str(error))

    async def _save_audio(
        self,
        text: str,
        voice: str,
        audio_path: Path,
    ) -> None:
        communicate = edge_tts.Communicate(
            text=text,
            voice=voice,
            rate=self.config.tts_edge_rate,
            volume=self.config.tts_edge_volume,
            pitch=self.config.tts_edge_pitch,
        )

        await communicate.save(str(audio_path))

    def speak(
        self,
        text: str,
        voice: str,
    ) -> EdgeTTSResult:
        result = self.synthesize_to_audio(text, voice)

        if not result.success or result.audio_path is None:
            return result

        try:
            play_result = AudioPlayer(self.config.tts_output_player).play(
                result.audio_path
            )

            return EdgeTTSResult(play_result.success, play_result.message)

        finally:
            try:
                Path(result.audio_path).unlink(missing_ok=True)
            except OSError:
                pass