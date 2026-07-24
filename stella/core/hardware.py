from __future__ import annotations

import os
import re
import shutil
import subprocess
from dataclasses import dataclass


@dataclass
class HardwareInfo:
    cpu_name: str
    cpu_cores: int
    ram_gb: float
    gpu_name: str
    gpu_vram_gb: float | None

    def summary(self) -> str:
        vram = f"{self.gpu_vram_gb:.1f} GB" if self.gpu_vram_gb is not None else "unknown"
        return (
            f"CPU: {self.cpu_name}\n"
            f"Cores/threads: {self.cpu_cores}\n"
            f"RAM: {self.ram_gb:.1f} GB\n"
            f"GPU: {self.gpu_name}\n"
            f"VRAM: {vram}"
        )


def _read_cpu_name() -> str:
    try:
        with open("/proc/cpuinfo", "r", encoding="utf-8", errors="ignore") as file:
            for line in file:
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass

    return "Unknown CPU"


def _read_ram_gb() -> float:
    try:
        with open("/proc/meminfo", "r", encoding="utf-8", errors="ignore") as file:
            for line in file:
                if line.startswith("MemTotal:"):
                    kb = int(line.split()[1])
                    return kb / 1024 / 1024
    except OSError:
        pass

    return 0.0


def _read_nvidia_gpu() -> tuple[str, float | None] | None:
    if not shutil.which("nvidia-smi"):
        return None

    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    if result.returncode != 0 or not result.stdout.strip():
        return None

    first_line = result.stdout.strip().splitlines()[0]
    parts = [part.strip() for part in first_line.split(",")]

    if not parts:
        return None

    name = parts[0]

    vram_gb = None
    if len(parts) >= 2:
        try:
            vram_gb = float(parts[1]) / 1024
        except ValueError:
            vram_gb = None

    return name, vram_gb


def _read_lspci_gpu() -> str:
    if not shutil.which("lspci"):
        return "Unknown GPU"

    result = subprocess.run(
        ["lspci"],
        text=True,
        capture_output=True,
        check=False,
    )

    if result.returncode != 0:
        return "Unknown GPU"

    for line in result.stdout.splitlines():
        if re.search(r"vga|3d controller|display", line, re.IGNORECASE):
            return line.strip()

    return "Unknown GPU"


def detect_hardware() -> HardwareInfo:
    nvidia = _read_nvidia_gpu()

    if nvidia:
        gpu_name, gpu_vram_gb = nvidia
    else:
        gpu_name = _read_lspci_gpu()
        gpu_vram_gb = None

    return HardwareInfo(
        cpu_name=_read_cpu_name(),
        cpu_cores=os.cpu_count() or 1,
        ram_gb=_read_ram_gb(),
        gpu_name=gpu_name,
        gpu_vram_gb=gpu_vram_gb,
    )