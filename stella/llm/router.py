from __future__ import annotations

from stella.core.config import StellaConfig
from stella.llm.ollama_provider import LLMResult, OllamaProvider


class LLMRouter:
    def __init__(self, config: StellaConfig) -> None:
        self.config = config
        self.ollama = OllamaProvider(config)

    def ask(self, text: str) -> LLMResult:
        if self.config.local_provider != "ollama":
            return LLMResult(
                success=False,
                message=f"Local provider поки не підтримується: {self.config.local_provider}",
                provider=self.config.local_provider,
                model=self.config.local_model,
            )

        return self.ollama.ask(text)