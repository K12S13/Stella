from __future__ import annotations

from dataclasses import dataclass

from stella.core.hardware import HardwareInfo


@dataclass(frozen=True)
class LocalModelOption:
    id: str
    ollama_model: str
    title: str
    runtime_memory_gb: float
    recommended_vram_gb: float | None
    quality_score: int
    speed_score: int
    ukrainian_score: int
    use_case: str
    notes: str


LOCAL_MODEL_CATALOG: list[LocalModelOption] = [
    LocalModelOption(
        id="qwen25_15b",
        ollama_model="qwen2.5:1.5b",
        title="Qwen2.5 1.5B",
        runtime_memory_gb=2.0,
        recommended_vram_gb=2.0,
        quality_score=2,
        speed_score=5,
        ukrainian_score=3,
        use_case="eco",
        notes="Дуже легка модель. Добра для простих команд і коротких відповідей.",
    ),
    LocalModelOption(
        id="llama32_1b",
        ollama_model="llama3.2:1b",
        title="Llama 3.2 1B",
        runtime_memory_gb=2.0,
        recommended_vram_gb=2.0,
        quality_score=2,
        speed_score=5,
        ukrainian_score=2,
        use_case="eco",
        notes="Швидка tiny-модель. Краще для англійської/простих задач.",
    ),
    LocalModelOption(
        id="qwen25_3b",
        ollama_model="qwen2.5:3b",
        title="Qwen2.5 3B",
        runtime_memory_gb=4.0,
        recommended_vram_gb=4.0,
        quality_score=3,
        speed_score=4,
        ukrainian_score=4,
        use_case="balanced",
        notes="Balanced default для Stella. Нормальний компроміс якість/швидкість/памʼять.",
    ),
    LocalModelOption(
        id="llama32_3b",
        ollama_model="llama3.2:3b",
        title="Llama 3.2 3B",
        runtime_memory_gb=4.0,
        recommended_vram_gb=4.0,
        quality_score=3,
        speed_score=4,
        ukrainian_score=2,
        use_case="balanced",
        notes="Добра instruction-модель, але українська не основний сильний бік.",
    ),
    LocalModelOption(
        id="gemma3_4b",
        ollama_model="gemma3:4b",
        title="Gemma 3 4B",
        runtime_memory_gb=5.0,
        recommended_vram_gb=5.0,
        quality_score=4,
        speed_score=3,
        ukrainian_score=3,
        use_case="quality",
        notes="Трохи важча, часто краща для пояснень, але може бути повільнішою.",
    ),
    LocalModelOption(
        id="qwen25_7b",
        ollama_model="qwen2.5:7b",
        title="Qwen2.5 7B",
        runtime_memory_gb=8.0,
        recommended_vram_gb=7.0,
        quality_score=5,
        speed_score=2,
        ukrainian_score=4,
        use_case="quality",
        notes="Краща якість, але вже важча. Для 12GB VRAM норм, якщо не тримати багато зайвого.",
    ),
    LocalModelOption(
        id="qwen25_coder_3b",
        ollama_model="qwen2.5-coder:3b",
        title="Qwen2.5 Coder 3B",
        runtime_memory_gb=4.0,
        recommended_vram_gb=4.0,
        quality_score=3,
        speed_score=4,
        ukrainian_score=2,
        use_case="code",
        notes="Краще для коду, скриптів, команд і технічних задач.",
    ),
]


def recommend_models(
    hardware: HardwareInfo,
    memory_budget_gb: float,
) -> list[tuple[LocalModelOption, str]]:
    recommended: list[tuple[LocalModelOption, str, float]] = []

    for model in LOCAL_MODEL_CATALOG:
        fits_ram_budget = model.runtime_memory_gb <= memory_budget_gb
        fits_vram = (
            hardware.gpu_vram_gb is not None
            and model.recommended_vram_gb is not None
            and model.recommended_vram_gb <= hardware.gpu_vram_gb
        )

        if not fits_ram_budget and not fits_vram:
            reason = "важкувата для вибраного бюджету"
            score = model.quality_score * 2 + model.speed_score + model.ukrainian_score - 6
        else:
            reason = "підходить"
            score = model.quality_score * 2 + model.speed_score + model.ukrainian_score

        if model.use_case == "balanced":
            score += 2

        if memory_budget_gb <= 2.5 and model.use_case == "eco":
            score += 3

        if memory_budget_gb >= 7.0 and model.use_case == "quality":
            score += 3

        recommended.append((model, reason, score))

    recommended.sort(key=lambda item: item[2], reverse=True)

    return [(model, reason) for model, reason, _score in recommended]


def recommended_num_ctx(memory_budget_gb: float) -> int:
    if memory_budget_gb <= 2.5:
        return 1024

    if memory_budget_gb <= 4.5:
        return 2048

    if memory_budget_gb <= 7.0:
        return 4096

    return 6144