from __future__ import annotations

import json
import tempfile
import wave
from collections import Counter
from pathlib import Path

import numpy as np
import sounddevice as sd
from vosk import KaldiRecognizer, Model, SetLogLevel

from stella.core.config import load_config
from stella.stt.wake_profile import (
    load_wake_profile,
    normalize_wake_text,
    save_wake_profile,
)
from stella.stt.whisper_file_transcriber import transcribe_wav_with_whisper


def record_wav(seconds: float, sample_rate: int, device_index: int | None) -> Path:
    print(f"Recording {seconds:.1f}s... Say wake word clearly.")

    audio = sd.rec(
        int(seconds * sample_rate),
        samplerate=sample_rate,
        channels=1,
        dtype="int16",
        device=device_index,
    )
    sd.wait()

    energy = float(np.abs(audio).mean())
    peak = int(np.abs(audio).max())

    print(f"Audio energy: mean={energy:.1f}, peak={peak}")

    temp_file = tempfile.NamedTemporaryFile(
        suffix=".wav",
        prefix="stella_wake_train_",
        delete=False,
    )
    wav_path = Path(temp_file.name)
    temp_file.close()

    with wave.open(str(wav_path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio.tobytes())

    return wav_path


def vosk_transcribe(wav_path: Path, model_path: str, sample_rate: int) -> str:
    model = Model(model_path)
    recognizer = KaldiRecognizer(model, sample_rate)

    with wave.open(str(wav_path), "rb") as wav_file:
        data = wav_file.readframes(wav_file.getnframes())

    recognizer.AcceptWaveform(data)
    payload = json.loads(recognizer.FinalResult())

    return normalize_wake_text(payload.get("text", ""))


def main() -> None:
    SetLogLevel(-1)

    config = load_config()

    print("")
    print("Stella wake word calibration")
    print("============================")
    print(f"Assistant name: {config.assistant_name}")
    print("")
    print("You will record the wake word several times.")
    print("Say ONLY the wake word, for example: Стелла")
    print("")

    samples_count_raw = input("Samples count [8]: ").strip()
    samples_count = int(samples_count_raw) if samples_count_raw else 8

    seconds_raw = input("Seconds per sample [1.4]: ").strip()
    seconds = float(seconds_raw) if seconds_raw else 1.4

    vosk_results: list[str] = []
    whisper_results: list[str] = []

    for index in range(samples_count):
        input(f"\nSample {index + 1}/{samples_count}. Press Enter and say wake word...")
        wav_path = record_wav(
            seconds=seconds,
            sample_rate=config.stt_sample_rate,
            device_index=config.stt_device_index,
        )

        try:
            vosk_text = vosk_transcribe(
                wav_path=wav_path,
                model_path=config.stt_vosk_model_path,
                sample_rate=config.stt_sample_rate,
            )

            whisper_result = transcribe_wav_with_whisper(config, wav_path)
            whisper_text = normalize_wake_text(
                whisper_result.text if whisper_result.success else ""
            )

            print(f"Vosk:    {vosk_text or '<empty>'}")
            print(f"Whisper: {whisper_text or '<empty>'}")

            if vosk_text:
                vosk_results.append(vosk_text)

            if whisper_text:
                whisper_results.append(whisper_text)

        finally:
            wav_path.unlink(missing_ok=True)

    print("")
    print("Detected variants")
    print("=================")

    vosk_counter = Counter(vosk_results)
    whisper_counter = Counter(whisper_results)

    print("Vosk:")
    for text, count in vosk_counter.most_common():
        print(f"  {text}: {count}")

    print("Whisper:")
    for text, count in whisper_counter.most_common():
        print(f"  {text}: {count}")

    base_aliases = [
        config.assistant_name,
        "стелла",
        "стела",
        "stella",
    ]

    detected_aliases = [
        text
        for text, count in vosk_counter.most_common()
        if text and count >= 1
    ]

    verification_aliases = [
        text
        for text, count in whisper_counter.most_common()
        if text and count >= 1
    ]

    aliases = []
    for item in [*base_aliases, *detected_aliases]:
        normalized = normalize_wake_text(item)

        if normalized and normalized not in aliases:
            aliases.append(normalized)

    verification = []
    for item in [*base_aliases, *verification_aliases]:
        normalized = normalize_wake_text(item)

        if normalized and normalized not in verification:
            verification.append(normalized)

    print("")
    print("Proposed wake aliases:")
    for item in aliases:
        print(f"  - {item}")

    print("")
    print("Proposed verification aliases:")
    for item in verification:
        print(f"  - {item}")

    accept = input("\nSave this wake profile? [Y/n]: ").strip().lower()

    if accept in {"n", "no", "ні"}:
        print("Cancelled.")
        return

    risky = input(
        "Allow calibrated alias if Whisper drops the wake word? "
        "This is faster but less strict. [Y/n]: "
    ).strip().lower()

    profile = load_wake_profile()
    profile["assistant_name"] = config.assistant_name
    profile["aliases"] = aliases
    profile["verification_aliases"] = verification
    profile["blocked_aliases"] = profile.get("blocked_aliases", [])
    profile["accept_alias_without_whisper_wake"] = risky not in {"n", "no", "ні"}

    save_wake_profile(profile)

    print("")
    print("Saved: stella/data/wake_profile.yaml")
    print("Restart Stella.")


if __name__ == "__main__":
    main()