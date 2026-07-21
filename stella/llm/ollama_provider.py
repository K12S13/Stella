from __future__ import annotations

from dataclasses import dataclass

import httpx

from stella.core.config import StellaConfig


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

    def ask(self, user_text: str) -> LLMResult:
        system_prompt = (
            "Ти Stella — локальний desktop assistant. "
            "Відповідай українською. "
            "Відповідай коротко: 1–5 речень. "
            "Не вигадуй виконання дій на комп’ютері. "
            "Якщо користувач просить відкрити, закрити або змінити щось у системі — скажи, що це має робити command router."
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