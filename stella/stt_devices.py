from __future__ import annotations

from stella.stt.providers.vosk_provider import list_input_devices


def main() -> None:
    devices = list_input_devices()

    if not devices:
        print("No input devices found.")
        return

    print("Input devices:")
    for device in devices:
        print(device)


if __name__ == "__main__":
    main()