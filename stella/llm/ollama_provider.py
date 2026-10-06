from __future__ import annotations

from dataclasses import dataclass

import httpx

from stella.core.config import StellaConfig

import json
from collections.abc import Iterator

import requests


@dataclass
class LLMResult:
    success: bool
    message: str
    provider: str = "ollama"
    model: str | None = None


class OllamaProvider:
    def __init__(self, config: StellaConfig) -> None:
        self.config = config
        self.base_url = config.ollama_base_url
        self.model = config.local_model

    def stream(self, text: str) -> Iterator[str]:
        url = f"{self.config.ollama_base_url}/api/chat"

        payload = {
            "model": self.config.local_model,
            "stream": True,
            "keep_alive": self.config.ollama_keep_alive,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Ти Stella — локальний desktop assistant. "
                        "Відповідай українською. "
                        "Відповідай коротко і практично. "
                        "Не вигадуй виконання дій на компʼютері."
                    ),
                },
                {
                    "role": "user",
                    "content": text,
                },
            ],
            "options": {
                "num_ctx": self.config.ollama_num_ctx,
                "temperature": 0.2,
                "top_p": 0.9,
            },
        }

        with requests.post(
            url,
            json=payload,
            stream=True,
            timeout=(5, self.config.llm_timeout_seconds),
        ) as response:
            response.raise_for_status()

            for raw_line in response.iter_lines(decode_unicode=True):
                if not raw_line:
                    continue

                data = json.loads(raw_line)

                if data.get("done"):
                    break

                chunk = data.get("message", {}).get("content", "")

                if chunk:
                    yield chunk

    def ask(self, user_text: str) -> LLMResult:
        system_prompt = (
            "Ти Stella — локальний desktop assistant. "
            "Відповідай українською. "
            "Відповідай коротко: 1–5 речень. "
            "Не вигадуй виконання дій на комп’ютері. "
            "Якщо користувач просить відкрити, закрити або змінити щось у системі — скажи, що це має робити command router."
            "Не використовуй китайські, японські або корейські символи. "
            "Не змішуй латиницю всередині українських або російських слів. "
            "Англійські технічні терміни пиши окремими англійськими словами. "
            "Якщо відповідь українською — кириличні слова мають бути кирилицею. "
        )

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_text},
            ],
            "stream": False,
            "keep_alive": self.config.ollama_keep_alive,
            "options": {
                "num_ctx": self.config.ollama_num_ctx,
                "temperature": 0.4,
                "top_p": 0.9,
            },
        }

        try:
            with httpx.Client(timeout=self.config.llm_timeout_seconds) as client:
                response = client.post(f"{self.base_url}/api/chat", json=payload)
                response.raise_for_status()
                data = response.json()

            message = data.get("message", {}).get("content", "").strip()

            if not message:
                return LLMResult(
                    success=False,
                    message="Ollama повернула порожню відповідь.",
                    model=self.model,
                )

            return LLMResult(
                success=True,
                message=message,
                model=self.model,
            )

        except httpx.ConnectError:
            return LLMResult(
                success=False,
                message="Не можу підключитися до Ollama. Перевір: systemctl status ollama",
                model=self.model,
            )
        except httpx.TimeoutException:
            return LLMResult(
                success=False,
                message="Ollama відповідає занадто довго. Можливо, модель важка або не вивантажилась.",
                model=self.model,
            )
        except Exception as error:
            return LLMResult(
                success=False,
                message=f"Помилка Ollama: {error}",
                model=self.model,
            )