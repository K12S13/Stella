from __future__ import annotations

from stella.core.config import load_config
from stella.tts.router import TTSRouter


def main() -> None:
    config = load_config()
    tts = TTSRouter(config)

    text = "привіт. я стелла. голосовий модуль працює."

    print(f"Speaking: {text}")
    result = tts.speak(text)

    print(result)


if __name__ == "__main__":
    main()