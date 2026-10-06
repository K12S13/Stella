from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QObject, Qt, QThread, Signal, QTimer
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFrame,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QStackedWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)
from stella.stt.wake_calibration_worker import WakeCalibrationWorker
from stella.stt.wake_profile import load_wake_profile

from stella.core.hardware import detect_hardware
from stella.core.model_catalog import (
    LOCAL_MODEL_CATALOG,
    recommended_num_ctx,
    recommend_models,
)
from stella.core.settings_store import SettingsStore
from stella.llm.ollama_manager import OllamaManager
from stella.tts.voice_catalog import (
    PIPER_VOICE_CATALOG,
    download_piper_voice,
    get_piper_voice,
)


class DownloadWorker(QObject):
    finished = Signal(bool, str, str)

    def __init__(self, kind: str, value: str) -> None:
        super().__init__()
        self.kind = kind
        self.value = value

    def run(self) -> None:
        if self.kind == "ollama":
            ok, message = OllamaManager().pull_model(self.value)
            self.finished.emit(ok, message, self.value)
            return

        if self.kind == "piper_voice":
            voice = get_piper_voice(self.value)

            if voice is None:
                self.finished.emit(False, f"Unknown Piper voice: {self.value}", "")
                return

            ok, message = download_piper_voice(voice)
            self.finished.emit(ok, message, voice.model_path)
            return

        self.finished.emit(False, f"Unknown download kind: {self.kind}", "")


class VoiceTestWorker(QObject):
    finished = Signal(bool, str, str)
    # ok, message, kind

    def __init__(self, kind: str) -> None:
        super().__init__()
        self.kind = kind

    def run(self) -> None:
        try:
            if self.kind == "tts":
                from stella.core.config import load_config
                from stella.tts.router import TTSRouter

                config = load_config()
                result = TTSRouter(config).speak(
                    "Привіт. Я Стелла. Голосовий модуль працює."
                )

                self.finished.emit(result.success, result.message, self.kind)
                return

            if self.kind == "stt":
                from stella.core.config import load_config
                from stella.stt.router import STTRouter

                config = load_config()
                result = STTRouter(config).listen_once()

                if result.success:
                    self.finished.emit(True, result.text, self.kind)
                else:
                    self.finished.emit(False, result.error or "STT failed.", self.kind)

                return

            self.finished.emit(False, f"Unknown voice test kind: {self.kind}", self.kind)

        except Exception as error:
            self.finished.emit(False, str(error), self.kind)

class SettingsWindow(QWidget):
    def __init__(self, parent=None, on_saved=None) -> None:
        super().__init__(parent)

        self.on_saved = on_saved
        self.store = SettingsStore()
        self.data = self.store.load_raw()
        self.hardware = detect_hardware()
        self.ollama = OllamaManager()

        self.download_thread: QThread | None = None
        self.download_worker: DownloadWorker | None = None
        self.voice_test_thread: QThread | None = None
        self.voice_test_worker: VoiceTestWorker | None = None
        self.wake_calibration_thread: QThread | None = None
        self.wake_calibration_worker: WakeCalibrationWorker | None = None

        self.setWindowTitle("Stella Control Center")
        self.resize(1040, 720)

        self.create_fields()
        self.build_layout()
        self.connect_live_updates()

        self.apply_style()
        self.refresh_recommendations()
        self.refresh_installed_models()
        self.select_current_voice()
        self.update_summary()
    def wake_profile_summary(self) -> str:
        profile = load_wake_profile()

        aliases = profile.get("aliases", [])
        verification_aliases = profile.get("verification_aliases", [])

        if not aliases and not verification_aliases:
            return "Wake profile: not calibrated"

        return (
            "Wake profile calibrated\n"
            f"Aliases: {', '.join(aliases) or 'none'}\n"
            f"Verification: {', '.join(verification_aliases) or 'none'}"
        )

    def create_fields(self) -> None:
        self.assistant_name = QLineEdit(self._get("assistant.name", "стелла"))

        self.wake_samples_count = QSpinBox()
        self.wake_samples_count.setRange(3, 20)
        self.wake_samples_count.setValue(8)

        self.wake_calibration_button = QPushButton("Calibrate wake word")
        self.wake_calibration_button.clicked.connect(self.start_wake_calibration)

        self.wake_calibration_status = QLabel(self.wake_profile_summary())
        self.wake_calibration_status.setWordWrap(True)

        self.wake_calibration_log = QTextEdit()
        self.wake_calibration_log.setReadOnly(True)
        self.wake_calibration_log.setFixedHeight(130)
        self.wake_calibration_log.setPlaceholderText("Calibration samples will appear here...")

        self.resource_mode = QComboBox()
        self.resource_mode.addItems(["eco", "balanced", "strong", "offline"])
        self.resource_mode.setCurrentText(self._get("modes.resource_mode", "balanced"))

        self.ram_budget = QSpinBox()
        self.ram_budget.setRange(1, max(2, int(self.hardware.ram_gb)))
        self.ram_budget.setSuffix(" GB")
        self.ram_budget.setValue(
            int(
                self._get(
                    "llm.local_model_memory_gb",
                    min(4, max(1, int(self.hardware.ram_gb) - 2)),
                )
            )
        )

        detected_vram = int(self.hardware.gpu_vram_gb or 4)

        self.vram_budget = QSpinBox()
        self.vram_budget.setRange(1, max(2, detected_vram))
        self.vram_budget.setSuffix(" GB")
        self.vram_budget.setValue(
            int(self._get("llm.local_model_vram_gb", min(8, max(1, detected_vram))))
        )

        self.cpu_threads = QSpinBox()
        self.cpu_threads.setRange(1, max(1, self.hardware.cpu_cores))
        self.cpu_threads.setSuffix(" threads")
        self.cpu_threads.setValue(
            int(self._get("llm.cpu_threads", min(4, self.hardware.cpu_cores)))
        )

        self.local_model = QLineEdit(self._get("llm.local_model", "qwen2.5:3b"))

        self.num_ctx = QSpinBox()
        self.num_ctx.setRange(512, 8192)
        self.num_ctx.setSingleStep(512)
        self.num_ctx.setValue(int(self._get("llm.num_ctx", 2048)))

        self.model_combo = QComboBox()
        self.model_notes = QTextEdit()
        self.model_notes.setReadOnly(True)
        self.model_notes.setMinimumHeight(150)

        self.installed_models = QTextEdit()
        self.installed_models.setReadOnly(True)
        self.installed_models.setMinimumHeight(110)

        self.tts_enabled = QCheckBox("Enable TTS")
        self.tts_enabled.setChecked(bool(self._get("tts.enabled", False)))

        self.speak_llm = QCheckBox("Speak LLM answers")
        self.speak_llm.setChecked(bool(self._get("tts.speak_llm_answers", True)))

        self.speak_commands = QCheckBox("Speak command results")
        self.speak_commands.setChecked(bool(self._get("tts.speak_command_results", False)))

        self.tts_model = QLineEdit(
            self._get(
                "tts.piper_model",
                "models/piper/uk_UA/ukrainian_tts/medium/uk_UA-ukrainian_tts-medium.onnx",
            )
        )

        self.output_player = QLineEdit(self._get("tts.output_player", "pw-play"))

        self.voice_combo = QComboBox()
        for voice in PIPER_VOICE_CATALOG:
            installed = "installed" if Path(voice.model_path).exists() else "not installed"
            self.voice_combo.addItem(
                f"{voice.title} · {voice.quality} · {installed}",
                voice.id,
            )

        self.tts_language_routing = QCheckBox("Use language-aware voice routing")
        self.tts_language_routing.setChecked(
            bool(self._get("tts.language_routing", True))
        )

        self.tts_default_language = QComboBox()
        self.tts_default_language.addItems(["uk", "ru", "en"])
        self.tts_default_language.setCurrentText(
            self._get("tts.default_language", "uk")
        )

        self.tts_uk_provider = QComboBox()
        self.tts_uk_provider.addItems(["edge", "piper", "silero", "rhvoice", "xtts", "mms"])
        self.tts_uk_provider.setCurrentText(self._get("tts.uk.provider", "piper"))

        self.tts_en_provider = QComboBox()
        self.tts_en_provider.addItems(["edge", "piper", "silero", "rhvoice", "xtts", "mms"])
        self.tts_en_provider.setCurrentText(self._get("tts.en.provider", "piper"))

        self.tts_ru_provider = QComboBox()
        self.tts_ru_provider.addItems(["edge", "piper", "silero", "rhvoice", "xtts", "mms"])
        self.tts_ru_provider.setCurrentText(self._get("tts.ru.provider", "piper"))

        self.tts_uk_voice = self.voice_combo_for_language(
            "uk",
            self._get(
                "tts.uk.model",
                "models/piper/uk_UA/tetiana/high/uk_UA-tetiana-high.onnx",
            ),
        )

        self.tts_en_voice = self.voice_combo_for_language(
            "en",
            self._get(
                "tts.en.model",
                "models/piper/en_US/amy/medium/en_US-amy-medium.onnx",
            ),
        )

        self.tts_ru_voice = self.voice_combo_for_language(
            "ru",
            self._get(
                "tts.ru.model",
                "models/piper/ru_RU/irina/medium/ru_RU-irina-medium.onnx",
            ),
        )
        self.tts_fallback_provider = QComboBox()
        self.tts_fallback_provider.addItems(["piper"])
        self.tts_fallback_provider.setCurrentText(
            self._get("tts.fallback_provider", "piper")
        )

        self.tts_edge_uk_voice = QComboBox()
        self.tts_edge_uk_voice.addItems([
            "uk-UA-PolinaNeural",
            "uk-UA-OstapNeural",
        ])
        self.tts_edge_uk_voice.setCurrentText(
            self._get("tts.edge.uk_voice", "uk-UA-PolinaNeural")
        )

        self.tts_edge_en_voice = QComboBox()
        self.tts_edge_en_voice.addItems([
            "en-US-JennyNeural",
            "en-US-AriaNeural",
            "en-US-GuyNeural",
            "en-GB-SoniaNeural",
            "en-GB-RyanNeural",
        ])
        self.tts_edge_en_voice.setCurrentText(
            self._get("tts.edge.en_voice", "en-US-JennyNeural")
        )

        self.tts_edge_ru_voice = QComboBox()
        self.tts_edge_ru_voice.addItems([
            "ru-RU-SvetlanaNeural",
            "ru-RU-DmitryNeural",
        ])
        self.tts_edge_ru_voice.setCurrentText(
            self._get("tts.edge.ru_voice", "ru-RU-SvetlanaNeural")
        )

        self.tts_edge_rate = QLineEdit(self._get("tts.edge.rate", "+0%"))
        self.tts_edge_volume = QLineEdit(self._get("tts.edge.volume", "+0%"))
        self.tts_edge_pitch = QLineEdit(self._get("tts.edge.pitch", "+0Hz"))

        self.confirm_dangerous = QCheckBox("Confirm dangerous actions")
        self.confirm_dangerous.setChecked(
            bool(self._get("safety.confirm_dangerous_actions", True))
        )

        self.allow_shell = QCheckBox("Allow shell commands")
        self.allow_shell.setChecked(bool(self._get("safety.allow_shell_commands", True)))

        self.status_label = QLabel("Ready")

        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.hide()

        self.summary_label = QLabel("")
        self.summary_label.setWordWrap(True)
        self.summary_label.setAlignment(Qt.AlignmentFlag.AlignTop)

    def voice_combo_for_language(
        self,
        language: str,
        current_model_path: str,
    ) -> QComboBox:
        combo = QComboBox()

        for voice in PIPER_VOICE_CATALOG:
            if voice.language != language:
                continue

            installed = "installed" if Path(voice.model_path).exists() else "not installed"
            combo.addItem(
                f"{voice.title} · {voice.quality} · {installed}",
                voice.model_path,
            )

        for index in range(combo.count()):
            if combo.itemData(index) == current_model_path:
                combo.setCurrentIndex(index)
                break

        return combo

    def build_layout(self) -> None:
        root = QVBoxLayout()

        header = self.build_header()
        body = QHBoxLayout()
        footer = self.build_footer()

        self.sidebar = QListWidget()
        self.sidebar.addItems(["System", "Models", "Voice", "Safety"])
        self.sidebar.setFixedWidth(150)
        self.sidebar.setCurrentRow(0)

        self.stack = QStackedWidget()
        self.stack.addWidget(self.build_system_page())
        self.stack.addWidget(self.build_models_page())
        self.stack.addWidget(self.build_voice_page())
        self.stack.addWidget(self.build_safety_page())

        self.sidebar.currentRowChanged.connect(self.stack.setCurrentIndex)

        body.addWidget(self.sidebar)
        body.addWidget(self.stack, 1)
        body.addWidget(self.build_summary_panel())

        root.addLayout(header)
        root.addLayout(body, 1)
        root.addLayout(footer)

        self.setLayout(root)

    def build_header(self) -> QHBoxLayout:
        layout = QHBoxLayout()

        title = QLabel("Stella Control Center")
        title.setObjectName("HeaderTitle")

        subtitle = QLabel("Local AI, hardware budget, models and voice configuration")
        subtitle.setObjectName("HeaderSubtitle")

        text_block = QVBoxLayout()
        text_block.addWidget(title)
        text_block.addWidget(subtitle)

        layout.addLayout(text_block)
        layout.addStretch()
        layout.addWidget(self.status_label)

        return layout

    def build_footer(self) -> QHBoxLayout:
        layout = QHBoxLayout()

        self.apply_button = QPushButton("Apply runtime")
        self.apply_button.clicked.connect(self.apply_runtime)

        self.save_button = QPushButton("Save settings")
        self.save_button.clicked.connect(self.save_settings)

        layout.addWidget(self.progress)
        layout.addStretch()
        layout.addWidget(self.apply_button)
        layout.addWidget(self.save_button)

        return layout

    def build_system_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout()

        layout.addWidget(self.page_title("System"))

        hardware_cards = QHBoxLayout()
        hardware_cards.addWidget(
            self.info_card(
                "CPU",
                f"{self.hardware.cpu_name}\n{self.hardware.cpu_cores} threads",
            )
        )
        hardware_cards.addWidget(
            self.info_card(
                "RAM",
                f"{self.hardware.ram_gb:.1f} GB total\nBudget: {self.ram_budget.value()} GB",
            )
        )
        hardware_cards.addWidget(
            self.info_card(
                "GPU",
                f"{self.hardware.gpu_name}\nVRAM: {self.hardware.gpu_vram_gb or 'unknown'} GB",
            )
        )

        layout.addLayout(hardware_cards)

        budget_group = QGroupBox("AI resource budget")
        budget_form = QFormLayout()
        budget_form.addRow("RAM for local models:", self.ram_budget)
        budget_form.addRow("VRAM for local models:", self.vram_budget)
        budget_form.addRow("CPU threads:", self.cpu_threads)
        budget_form.addRow("Resource mode:", self.resource_mode)
        budget_group.setLayout(budget_form)

        assistant_group = QGroupBox("Assistant")
        assistant_form = QFormLayout()
        assistant_form.addRow("Assistant name:", self.assistant_name)
        wake_group = QGroupBox("Wake word calibration")
        wake_form = QFormLayout()

        wake_form.addRow("Samples:", self.wake_samples_count)
        wake_form.addRow("Calibration:", self.wake_calibration_button)
        wake_form.addRow("Status:", self.wake_calibration_status)
        wake_form.addRow("Log:", self.wake_calibration_log)

        wake_group.setLayout(wake_form)
        layout.addWidget(wake_group)
        assistant_group.setLayout(assistant_form)

        layout.addWidget(budget_group)
        layout.addWidget(assistant_group)
        layout.addStretch()

        page.setLayout(layout)
        return page

    def start_wake_calibration(self) -> None:
        if self.wake_calibration_thread is not None:
            QMessageBox.warning(
                self,
                "Wake calibration",
                "Калібровка вже запущена.",
            )
            return

        self.save_settings(show_message=False)

        assistant_name = self.assistant_name.text().strip()

        if not assistant_name:
            QMessageBox.warning(
                self,
                "Wake calibration",
                "Спочатку введи назву системи.",
            )
            return

        parent = self.parent()

        if parent is not None and hasattr(parent, "stop_wake_listener"):
            parent.stop_wake_listener()

        self.wake_calibration_log.clear()
        self.wake_calibration_status.setText(
            f"Calibration started. Say only: {assistant_name}"
        )

        self.wake_calibration_button.setEnabled(False)

        self.progress.show()
        self.status_label.setText("Wake calibration running...")

        config = load_config()

        self.wake_calibration_thread = QThread()
        self.wake_calibration_worker = WakeCalibrationWorker(
            config=config,
            assistant_name=assistant_name,
            samples_count=int(self.wake_samples_count.value()),
            seconds_per_sample=1.2,
        )

        self.wake_calibration_worker.moveToThread(self.wake_calibration_thread)

        self.wake_calibration_thread.started.connect(
            self.wake_calibration_worker.run
        )

        self.wake_calibration_worker.status.connect(
            self.on_wake_calibration_status
        )

        self.wake_calibration_worker.sample_result.connect(
            self.on_wake_calibration_sample
        )

        self.wake_calibration_worker.finished.connect(
            self.on_wake_calibration_finished
        )

        self.wake_calibration_worker.finished.connect(
            self.wake_calibration_thread.quit
        )

        self.wake_calibration_worker.finished.connect(
            self.wake_calibration_worker.deleteLater
        )

        self.wake_calibration_thread.finished.connect(
            self.wake_calibration_thread.deleteLater
        )

        self.wake_calibration_thread.finished.connect(
            self.cleanup_wake_calibration_refs
        )

        self.wake_calibration_thread.start()
        QTimer.singleShot(45000, self.fail_stuck_wake_calibration)

    def fail_stuck_wake_calibration(self) -> None:
        if self.wake_calibration_thread is None:
            return

        if not self.wake_calibration_thread.isRunning():
            return

        if self.wake_calibration_worker is not None:
            self.wake_calibration_worker.cancel()

        self.progress.hide()
        self.wake_calibration_button.setEnabled(True)
        self.status_label.setText("Wake calibration timeout.")
        self.wake_calibration_log.append("Error: calibration timeout after 45 seconds.")

    def on_wake_calibration_status(self, message: str) -> None:
        self.status_label.setText(message)

        if hasattr(self, "wake_calibration_log"):
            self.wake_calibration_log.append(f"Status: {message}")

    def on_wake_calibration_sample(
        self,
        index: int,
        vosk_text: str,
        whisper_text: str,
        mean_energy: float,
        peak: int,
    ) -> None:
        self.wake_calibration_log.append(
            f"Sample {index}: Vosk='{vosk_text}' | "
            f"energy={mean_energy:.1f} | peak={peak}"
        )

    def on_wake_calibration_finished(self, ok: bool, message: str) -> None:
        self.progress.hide()
        self.wake_calibration_button.setEnabled(True)

        self.wake_calibration_status.setText(self.wake_profile_summary())

        if ok:
            self.status_label.setText("Wake calibration saved.")
            self.wake_calibration_log.append(message)
            QMessageBox.information(
                self,
                "Wake calibration",
                message,
            )
        else:
            self.status_label.setText("Wake calibration failed.")
            self.wake_calibration_log.append(f"Error: {message}")
            QMessageBox.warning(
                self,
                "Wake calibration failed",
                message[-1500:],
            )

    def cleanup_wake_calibration_refs(self) -> None:
        self.wake_calibration_thread = None
        self.wake_calibration_worker = None

    def build_models_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout()

        layout.addWidget(self.page_title("Local AI Models"))

        current_group = QGroupBox("Current model")
        current_form = QFormLayout()
        current_form.addRow("Ollama model:", self.local_model)
        current_form.addRow("Context size:", self.num_ctx)
        current_group.setLayout(current_form)

        recommended_group = QGroupBox("Recommended for this PC")
        recommended_layout = QVBoxLayout()
        recommended_layout.addWidget(self.model_combo)
        recommended_layout.addWidget(self.model_notes)

        model_buttons = QHBoxLayout()

        self.use_model_button = QPushButton("Use selected")
        self.use_model_button.clicked.connect(self.use_selected_model)

        self.download_model_button = QPushButton("Download selected")
        self.download_model_button.clicked.connect(self.download_selected_model)

        self.refresh_models_button = QPushButton("Refresh installed")
        self.refresh_models_button.clicked.connect(self.refresh_installed_models)

        model_buttons.addWidget(self.use_model_button)
        model_buttons.addWidget(self.download_model_button)
        model_buttons.addWidget(self.refresh_models_button)

        recommended_layout.addLayout(model_buttons)
        recommended_group.setLayout(recommended_layout)

        installed_group = QGroupBox("Installed Ollama models")
        installed_layout = QVBoxLayout()
        installed_layout.addWidget(self.installed_models)
        installed_group.setLayout(installed_layout)

        layout.addWidget(current_group)
        layout.addWidget(recommended_group)
        layout.addWidget(installed_group)

        page.setLayout(layout)
        return page

    def build_voice_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout()

        layout.addWidget(self.page_title("Voice / STT / TTS"))

        # -------------------------
        # TTS
        # -------------------------
        tts_group = QGroupBox("Text-to-Speech")
        tts_form = QFormLayout()

        tts_form.addRow("", self.tts_enabled)
        tts_form.addRow("", self.speak_llm)
        tts_form.addRow("", self.speak_commands)
        tts_form.addRow("Piper model path:", self.tts_model)
        tts_form.addRow("Output player:", self.output_player)

        self.test_tts_button = QPushButton("Test TTS")
        self.test_tts_button.clicked.connect(lambda: self.start_voice_test_worker("tts"))
        tts_form.addRow("Test voice output:", self.test_tts_button)

        tts_group.setLayout(tts_form)

        # -------------------------
        # Voice catalog
        # -------------------------
        voice_group = QGroupBox("Piper voice catalog")
        voice_layout = QVBoxLayout()
        voice_layout.addWidget(self.voice_combo)

        voice_buttons = QHBoxLayout()

        self.use_voice_button = QPushButton("Use selected voice")
        self.use_voice_button.clicked.connect(self.use_selected_voice)

        self.download_voice_button = QPushButton("Download selected voice")
        self.download_voice_button.clicked.connect(self.download_selected_voice)

        voice_buttons.addWidget(self.use_voice_button)
        voice_buttons.addWidget(self.download_voice_button)

        voice_layout.addLayout(voice_buttons)
        voice_group.setLayout(voice_layout)

        routing_group = QGroupBox("Language-aware voice routing")
        routing_form = QFormLayout()

        routing_form.addRow("", self.tts_language_routing)
        routing_form.addRow("Default language:", self.tts_default_language)

        routing_form.addRow("Ukrainian provider:", self.tts_uk_provider)
        routing_form.addRow("Ukrainian voice:", self.tts_uk_voice)

        routing_form.addRow("English provider:", self.tts_en_provider)
        routing_form.addRow("English voice:", self.tts_en_voice)

        routing_form.addRow("Russian provider:", self.tts_ru_provider)
        routing_form.addRow("Russian voice:", self.tts_ru_voice)

        routing_group.setLayout(routing_form)

        edge_group = QGroupBox("Edge TTS online voices")
        edge_form = QFormLayout()

        edge_form.addRow("Fallback provider:", self.tts_fallback_provider)
        edge_form.addRow("Ukrainian Edge voice:", self.tts_edge_uk_voice)
        edge_form.addRow("English Edge voice:", self.tts_edge_en_voice)
        edge_form.addRow("Russian Edge voice:", self.tts_edge_ru_voice)
        edge_form.addRow("Rate:", self.tts_edge_rate)
        edge_form.addRow("Volume:", self.tts_edge_volume)
        edge_form.addRow("Pitch:", self.tts_edge_pitch)

        edge_group.setLayout(edge_form)


        # -------------------------
        # STT
        # -------------------------
        stt_group = QGroupBox("Speech-to-Text")
        stt_form = QFormLayout()

        self.stt_provider_combo = QComboBox()
        self.stt_provider_combo.addItems(["whisper", "vosk", "dummy", "none"])
        self.stt_provider_combo.setCurrentText(self._get("stt.provider", "whisper"))

        self.stt_auto_submit_checkbox = QCheckBox("Auto-submit після розпізнавання")
        self.stt_auto_submit_checkbox.setChecked(
            bool(self._get("stt.auto_submit", False))
        )

        self.stt_listen_seconds_spin = QSpinBox()
        self.stt_listen_seconds_spin.setRange(1, 180)
        self.stt_listen_seconds_spin.setValue(
            int(self._get("stt.listen_seconds", 60))
        )

        self.stt_sample_rate_combo = QComboBox()
        self.stt_sample_rate_combo.addItems(["16000", "48000"])
        self.stt_sample_rate_combo.setCurrentText(
            str(self._get("stt.sample_rate", 16000))
        )

        self.stt_device_index_input = QLineEdit()
        device_index = self._get("stt.device_index", None)

        if device_index is None:
            self.stt_device_index_input.setText("")
            self.stt_device_index_input.setPlaceholderText("default")
        else:
            self.stt_device_index_input.setText(str(device_index))

        self.vosk_model_path_input = QLineEdit()
        self.vosk_model_path_input.setText(
            self._get("stt.vosk_model_path", "models/vosk/uk-small")
        )

        self.whisper_model_combo = QComboBox()
        self.whisper_model_combo.addItems([
            "small",
            "medium",
            "large-v3-turbo",
            "large-v3",
        ])
        self.whisper_model_combo.setCurrentText(
            self._get("stt.whisper_model", "large-v3-turbo")
        )

        self.whisper_device_combo = QComboBox()
        self.whisper_device_combo.addItems(["cuda", "cpu", "auto"])
        self.whisper_device_combo.setCurrentText(
            self._get("stt.whisper_device", "cuda")
        )

        self.whisper_compute_combo = QComboBox()
        self.whisper_compute_combo.addItems([
            "int8_float16",
            "float16",
            "int8",
            "auto",
        ])
        self.whisper_compute_combo.setCurrentText(
            self._get("stt.whisper_compute_type", "int8_float16")
        )

        self.whisper_language_combo = QComboBox()
        self.whisper_language_combo.addItems(["uk", "ru", "en", "auto"])

        current_language = self._get("stt.whisper_language", "uk") or "auto"
        self.whisper_language_combo.setCurrentText(current_language)

        self.whisper_beam_spin = QSpinBox()
        self.whisper_beam_spin.setRange(1, 8)
        self.whisper_beam_spin.setValue(
            int(self._get("stt.whisper_beam_size", 1))
        )

        self.whisper_vad_checkbox = QCheckBox("VAD filter")
        self.whisper_vad_checkbox.setChecked(
            bool(self._get("stt.whisper_vad_filter", True))
        )

        self.test_stt_button = QPushButton("Test STT")
        self.test_stt_button.clicked.connect(lambda: self.start_voice_test_worker("stt"))

        stt_form.addRow("Provider:", self.stt_provider_combo)
        stt_form.addRow("Auto-submit:", self.stt_auto_submit_checkbox)
        stt_form.addRow("Max recording seconds:", self.stt_listen_seconds_spin)
        stt_form.addRow("Sample rate:", self.stt_sample_rate_combo)
        stt_form.addRow("Mic device index:", self.stt_device_index_input)
        stt_form.addRow("Vosk model path:", self.vosk_model_path_input)

        stt_form.addRow("Whisper model:", self.whisper_model_combo)
        stt_form.addRow("Whisper device:", self.whisper_device_combo)
        stt_form.addRow("Whisper compute:", self.whisper_compute_combo)
        stt_form.addRow("Whisper language:", self.whisper_language_combo)
        stt_form.addRow("Whisper beam size:", self.whisper_beam_spin)
        stt_form.addRow("Whisper VAD:", self.whisper_vad_checkbox)
        stt_form.addRow("Test microphone:", self.test_stt_button)

        stt_group.setLayout(stt_form)

        layout.addWidget(tts_group)
        layout.addWidget(voice_group)
        layout.addWidget(routing_group)
        layout.addWidget(edge_group)
        layout.addWidget(stt_group)

        page.setLayout(layout)
        return page

    def build_safety_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout()

        layout.addWidget(self.page_title("Safety"))

        safety_group = QGroupBox("Command safety")
        safety_layout = QVBoxLayout()
        safety_layout.addWidget(self.confirm_dangerous)
        safety_layout.addWidget(self.allow_shell)
        safety_group.setLayout(safety_layout)

        warning = QLabel(
            "Dangerous actions should require confirmation: closing apps, shell commands, "
            "file deletion, large downloads and system-level changes."
        )
        warning.setWordWrap(True)
        warning.setObjectName("WarningText")

        layout.addWidget(safety_group)
        layout.addWidget(warning)
        layout.addStretch()

        page.setLayout(layout)
        return page

    def build_summary_panel(self) -> QWidget:
        panel = QFrame()
        panel.setObjectName("SummaryPanel")
        panel.setFixedWidth(250)

        layout = QVBoxLayout()

        title = QLabel("Current setup")
        title.setObjectName("PanelTitle")

        hint = QLabel("Tests are inside Voice settings.")
        hint.setWordWrap(True)
        hint.setObjectName("HeaderSubtitle")

        layout.addWidget(title)
        layout.addWidget(self.summary_label)
        layout.addStretch()
        layout.addWidget(hint)

        panel.setLayout(layout)
        return panel

    def page_title(self, text: str) -> QLabel:
        label = QLabel(text)
        label.setObjectName("PageTitle")
        return label

    def info_card(self, title: str, body: str) -> QFrame:
        card = QFrame()
        card.setObjectName("InfoCard")

        layout = QVBoxLayout()

        title_label = QLabel(title)
        title_label.setObjectName("CardTitle")

        body_label = QLabel(body)
        body_label.setWordWrap(True)

        layout.addWidget(title_label)
        layout.addWidget(body_label)

        card.setLayout(layout)
        return card

    def connect_live_updates(self) -> None:
        self.ram_budget.valueChanged.connect(self.refresh_recommendations)
        self.ram_budget.valueChanged.connect(self.update_summary)

        self.vram_budget.valueChanged.connect(self.update_summary)
        self.cpu_threads.valueChanged.connect(self.update_summary)

        self.resource_mode.currentTextChanged.connect(self.update_summary)
        self.local_model.textChanged.connect(self.update_summary)
        self.tts_model.textChanged.connect(self.update_summary)
        self.tts_enabled.toggled.connect(self.update_summary)

        self.model_combo.currentIndexChanged.connect(self.on_model_selected)
        self.voice_combo.currentIndexChanged.connect(self.on_voice_selected)

        # STT widgets are created inside build_voice_page(),
        # so guard with hasattr for safe initialization order.
        if hasattr(self, "stt_provider_combo"):
            self.stt_provider_combo.currentTextChanged.connect(self.update_summary)

        if hasattr(self, "whisper_model_combo"):
            self.whisper_model_combo.currentTextChanged.connect(self.update_summary)

        if hasattr(self, "whisper_device_combo"):
            self.whisper_device_combo.currentTextChanged.connect(self.update_summary)

    def _get(self, path: str, default):
        current = self.data

        for part in path.split("."):
            if not isinstance(current, dict):
                return default

            current = current.get(part)

            if current is None:
                return default

        return current

    def refresh_recommendations(self, *_args) -> None:
        memory_budget = float(self.ram_budget.value())
        recommendations = recommend_models(self.hardware, memory_budget)

        current_model = self.local_model.text().strip()

        self.model_combo.blockSignals(True)
        self.model_combo.clear()

        for index, (model, reason) in enumerate(recommendations):
            status = "installed" if model.ollama_model in self.ollama.installed_models() else "not installed"

            label = (
                f"{index + 1}. {model.ollama_model} · "
                f"RAM~{model.runtime_memory_gb:.1f}GB · "
                f"Q{model.quality_score}/S{model.speed_score}/UK{model.ukrainian_score} · "
                f"{status} · {reason}"
            )

            self.model_combo.addItem(label, model.id)

            if model.ollama_model == current_model:
                self.model_combo.setCurrentIndex(index)

        self.model_combo.blockSignals(False)
        self.on_model_selected()
        self.update_summary()

    def refresh_installed_models(self) -> None:
        models = self.ollama.installed_models()

        if not models:
            self.installed_models.setText("No installed Ollama models detected.")
            return

        self.installed_models.setText("\n".join(models))

    def _selected_model(self):
        model_id = self.model_combo.currentData()

        for model in LOCAL_MODEL_CATALOG:
            if model.id == model_id:
                return model

        return None

    def on_model_selected(self, *_args) -> None:
        model = self._selected_model()

        if model is None:
            self.model_notes.setText("")
            return

        installed = model.ollama_model in self.ollama.installed_models()

        warning = ""
        if model.runtime_memory_gb > self.ram_budget.value():
            warning = "\nWarning: this model is heavier than selected RAM budget."

        self.model_notes.setText(
            f"Model: {model.title}\n"
            f"Ollama tag: {model.ollama_model}\n"
            f"Status: {'Installed' if installed else 'Not installed'}\n"
            f"Use case: {model.use_case}\n"
            f"Estimated runtime memory: ~{model.runtime_memory_gb:.1f} GB\n"
            f"Recommended VRAM: {model.recommended_vram_gb or 'unknown'} GB\n"
            f"Quality: {model.quality_score}/5\n"
            f"Speed: {model.speed_score}/5\n"
            f"Ukrainian: {model.ukrainian_score}/5\n"
            f"Notes: {model.notes}"
            f"{warning}"
        )

    def use_selected_model(self) -> None:
        model = self._selected_model()

        if model is None:
            return

        self.local_model.setText(model.ollama_model)
        self.num_ctx.setValue(recommended_num_ctx(float(self.ram_budget.value())))
        self.status_label.setText(f"Selected model: {model.ollama_model}")
        self.update_summary()

    def download_selected_model(self) -> None:
        model = self._selected_model()

        if model is None:
            return

        if model.runtime_memory_gb > self.ram_budget.value():
            confirm = QMessageBox.question(
                self,
                "Heavy model",
                (
                    f"{model.ollama_model} may be heavier than your selected budget.\n\n"
                    "Download anyway?"
                ),
            )

            if confirm != QMessageBox.StandardButton.Yes:
                return

        self.start_download_worker("ollama", model.ollama_model)

    def select_current_voice(self) -> None:
        current_path = self.tts_model.text().strip()

        for index, voice in enumerate(PIPER_VOICE_CATALOG):
            if voice.model_path == current_path:
                self.voice_combo.setCurrentIndex(index)
                return

    def _selected_voice(self):
        voice_id = self.voice_combo.currentData()
        return get_piper_voice(str(voice_id))

    def on_voice_selected(self, *_args) -> None:
        voice = self._selected_voice()

        if voice is None:
            return

        status = "installed" if Path(voice.model_path).exists() else "not installed"
        self.status_label.setText(f"Voice: {voice.title} · {voice.quality} · {status}")

    def use_selected_voice(self) -> None:
        voice = self._selected_voice()

        if voice is None:
            return

        self.tts_model.setText(voice.model_path)
        self.tts_enabled.setChecked(True)
        self.status_label.setText(f"Selected voice: {voice.title}")
        self.update_summary()

    def download_selected_voice(self) -> None:
        voice = self._selected_voice()

        if voice is None:
            return

        self.start_download_worker("piper_voice", voice.id)

    def start_voice_test_worker(self, kind: str) -> None:
        if self.voice_test_thread is not None:
            QMessageBox.warning(self, "Voice test active", "Тест уже виконується.")
            return

        # Save current UI values before testing.
        self.save_settings(show_message=False)

        self.status_label.setText(f"Running {kind.upper()} test...")
        self.progress.show()

        if hasattr(self, "test_tts_button"):
            self.test_tts_button.setEnabled(False)

        if hasattr(self, "test_stt_button"):
            self.test_stt_button.setEnabled(False)

        self.voice_test_thread = QThread()
        self.voice_test_worker = VoiceTestWorker(kind)

        self.voice_test_worker.moveToThread(self.voice_test_thread)

        self.voice_test_thread.started.connect(self.voice_test_worker.run)
        self.voice_test_worker.finished.connect(self.on_voice_test_finished)
        self.voice_test_worker.finished.connect(self.voice_test_thread.quit)
        self.voice_test_worker.finished.connect(self.voice_test_worker.deleteLater)

        self.voice_test_thread.finished.connect(self.voice_test_thread.deleteLater)
        self.voice_test_thread.finished.connect(self.cleanup_voice_test_refs)

        self.voice_test_thread.start()

    def on_voice_test_finished(self, ok: bool, message: str, kind: str) -> None:
        self.progress.hide()

        if hasattr(self, "test_tts_button"):
            self.test_tts_button.setEnabled(True)

        if hasattr(self, "test_stt_button"):
            self.test_stt_button.setEnabled(True)

        if ok:
            if kind == "stt":
                self.status_label.setText(f"STT recognized: {message}")
                QMessageBox.information(self, "STT test OK", f"Recognized:\n{message}")
            else:
                self.status_label.setText("TTS test OK.")
        else:
            self.status_label.setText(f"{kind.upper()} test failed.")
            QMessageBox.warning(self, f"{kind.upper()} test failed", message[-1500:])

    def cleanup_voice_test_refs(self) -> None:
        self.voice_test_thread = None
        self.voice_test_worker = None

    def start_download_worker(self, kind: str, value: str) -> None:
        if self.download_thread is not None:
            QMessageBox.warning(self, "Download active", "Завантаження вже йде.")
            return

        self.status_label.setText(f"Downloading: {value}")
        self.progress.show()
        self.download_model_button.setEnabled(False)
        self.download_voice_button.setEnabled(False)

        self.download_thread = QThread()
        self.download_worker = DownloadWorker(kind, value)

        self.download_worker.moveToThread(self.download_thread)

        self.download_thread.started.connect(self.download_worker.run)
        self.download_worker.finished.connect(self.on_download_finished)
        self.download_worker.finished.connect(self.download_thread.quit)
        self.download_worker.finished.connect(self.download_worker.deleteLater)

        self.download_thread.finished.connect(self.download_thread.deleteLater)
        self.download_thread.finished.connect(self.cleanup_download_refs)

        self.download_thread.start()

    def cleanup_download_refs(self) -> None:
        self.download_thread = None
        self.download_worker = None

    def on_download_finished(self, ok: bool, message: str, extra: str) -> None:
        self.progress.hide()
        self.download_model_button.setEnabled(True)
        self.download_voice_button.setEnabled(True)

        if not ok:
            self.status_label.setText("Download failed.")
            QMessageBox.warning(self, "Download failed", message[-1500:])
            return

        self.status_label.setText("Download OK.")

        if extra.endswith(".onnx"):
            self.tts_model.setText(extra)
            self.tts_enabled.setChecked(True)
            self.update_summary()
            QMessageBox.information(self, "Voice downloaded", f"Voice ready:\n{extra}")
            return

        self.refresh_installed_models()
        self.refresh_recommendations()

        use_now = QMessageBox.question(
            self,
            "Model downloaded",
            f"Model downloaded:\n{extra}\n\nUse it as default now?",
        )

        if use_now == QMessageBox.StandardButton.Yes:
            self.local_model.setText(extra)
            self.update_summary()

    def apply_runtime(self) -> None:
        self.save_settings(show_message=False)

        if self.on_saved:
            self.on_saved()

        self.status_label.setText("Runtime updated.")

    def save_settings(self, show_message: bool = True) -> None:
        assistant_name = self.assistant_name.text().strip() or "стелла"

        self.store.set_value("assistant.name", assistant_name)
        self.store.set_value("wake.words", [assistant_name])
        self.store.set_value("modes.resource_mode", self.resource_mode.currentText())

        self.store.set_value("llm.local_model_memory_gb", int(self.ram_budget.value()))
        self.store.set_value("llm.local_model_vram_gb", int(self.vram_budget.value()))
        self.store.set_value("llm.cpu_threads", int(self.cpu_threads.value()))
        self.store.set_value("llm.local_model", self.local_model.text().strip() or "qwen2.5:3b")
        self.store.set_value("llm.num_ctx", int(self.num_ctx.value()))

        self.store.set_value("tts.enabled", self.tts_enabled.isChecked())
        self.store.set_value("tts.provider", "piper")
        self.store.set_value("tts.speak_llm_answers", self.speak_llm.isChecked())
        self.store.set_value("tts.speak_command_results", self.speak_commands.isChecked())
        self.store.set_value("tts.piper_model", self.tts_model.text().strip())
        self.store.set_value("tts.output_player", self.output_player.text().strip() or "pw-play")
        self.store.set_value("tts.language_routing", self.tts_language_routing.isChecked())
        self.store.set_value("tts.default_language", self.tts_default_language.currentText())

        self.store.set_value("tts.uk.provider", self.tts_uk_provider.currentText())
        self.store.set_value("tts.uk.model", self.tts_uk_voice.currentData())

        self.store.set_value("tts.en.provider", self.tts_en_provider.currentText())
        self.store.set_value("tts.en.model", self.tts_en_voice.currentData())

        self.store.set_value("tts.ru.provider", self.tts_ru_provider.currentText())
        self.store.set_value("tts.ru.model", self.tts_ru_voice.currentData())

        self.store.set_value("tts.fallback_provider", self.tts_fallback_provider.currentText())

        self.store.set_value("tts.edge.uk_voice", self.tts_edge_uk_voice.currentText())
        self.store.set_value("tts.edge.en_voice", self.tts_edge_en_voice.currentText())
        self.store.set_value("tts.edge.ru_voice", self.tts_edge_ru_voice.currentText())

        self.store.set_value("tts.edge.rate", self.tts_edge_rate.text().strip() or "+0%")
        self.store.set_value("tts.edge.volume", self.tts_edge_volume.text().strip() or "+0%")
        self.store.set_value("tts.edge.pitch", self.tts_edge_pitch.text().strip() or "+0Hz")
        device_index_text = self.stt_device_index_input.text().strip()

        if device_index_text.lower() in {"", "none", "null"}:
            device_index = None
        else:
            device_index = int(device_index_text)

        whisper_language = self.whisper_language_combo.currentText()

        if whisper_language == "auto":
            whisper_language = ""

        self.store.set_value("stt.provider", self.stt_provider_combo.currentText())
        self.store.set_value("stt.auto_submit", self.stt_auto_submit_checkbox.isChecked())
        self.store.set_value("stt.listen_seconds", int(self.stt_listen_seconds_spin.value()))
        self.store.set_value("stt.sample_rate", int(self.stt_sample_rate_combo.currentText()))
        self.store.set_value("stt.device_index", device_index)
        self.store.set_value("stt.vosk_model_path", self.vosk_model_path_input.text().strip())

        self.store.set_value("stt.whisper_model", self.whisper_model_combo.currentText())
        self.store.set_value("stt.whisper_device", self.whisper_device_combo.currentText())
        self.store.set_value("stt.whisper_compute_type", self.whisper_compute_combo.currentText())
        self.store.set_value("stt.whisper_language", whisper_language)
        self.store.set_value("stt.whisper_beam_size", int(self.whisper_beam_spin.value()))
        self.store.set_value("stt.whisper_vad_filter", self.whisper_vad_checkbox.isChecked())

        self.store.set_value("safety.confirm_dangerous_actions", self.confirm_dangerous.isChecked())
        self.store.set_value("safety.allow_shell_commands", self.allow_shell.isChecked())

        self.status_label.setText("Settings saved.")
        self.update_summary()

        if show_message:
            QMessageBox.information(
                self,
                "Settings saved",
                "Налаштування збережено.",
            )

        if self.on_saved:
            self.on_saved()

    def update_summary(self, *_args) -> None:
        voice_name = "unknown"

        for voice in PIPER_VOICE_CATALOG:
            if voice.model_path == self.tts_model.text().strip():
                voice_name = voice.title
                break

        stt_provider = "unknown"
        stt_model = "unknown"
        stt_device = "unknown"

        if hasattr(self, "stt_provider_combo"):
            stt_provider = self.stt_provider_combo.currentText()

        if hasattr(self, "whisper_model_combo"):
            stt_model = self.whisper_model_combo.currentText()

        if hasattr(self, "whisper_device_combo"):
            stt_device = self.whisper_device_combo.currentText()

        self.summary_label.setText(
            f"Mode: {self.resource_mode.currentText()}\n\n"
            f"Model: {self.local_model.text().strip() or 'none'}\n"
            f"Context: {self.num_ctx.value()}\n\n"
            f"RAM budget: {self.ram_budget.value()} GB\n"
            f"VRAM budget: {self.vram_budget.value()} GB\n"
            f"CPU threads: {self.cpu_threads.value()}\n\n"
            f"STT: {stt_provider}\n"
            f"Whisper: {stt_model} / {stt_device}\n\n"
            f"TTS: {'enabled' if self.tts_enabled.isChecked() else 'disabled'}\n"
            f"Voice: {voice_name}\n\n"
            f"Ollama: {'available' if self.ollama.is_available() else 'not found'}\n"
            f"GPU: {self.hardware.gpu_name}"
        )

    def apply_style(self) -> None:
        self.setStyleSheet("""
            QWidget {
                background: #101010;
                color: #e8e8e8;
                font-size: 14px;
            }

            #HeaderTitle {
                font-size: 24px;
                font-weight: bold;
                color: #ffffff;
            }

            #HeaderSubtitle {
                color: #9f9f9f;
            }

            #PageTitle {
                font-size: 21px;
                font-weight: bold;
                margin-bottom: 8px;
            }

            QListWidget {
                background: #151515;
                border: 1px solid #2d2d2d;
                border-radius: 10px;
                padding: 6px;
            }

            QListWidget::item {
                padding: 10px;
                border-radius: 8px;
            }

            QListWidget::item:selected {
                background: #2d2d2d;
                color: #ffffff;
            }

            QGroupBox {
                border: 1px solid #333333;
                border-radius: 10px;
                margin-top: 12px;
                padding: 12px;
                font-weight: bold;
            }

            QGroupBox::title {
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 4px;
            }

            #InfoCard {
                background: #171717;
                border: 1px solid #333333;
                border-radius: 12px;
                padding: 8px;
            }

            #CardTitle {
                font-size: 16px;
                font-weight: bold;
                color: #ffffff;
            }

            #SummaryPanel {
                background: #151515;
                border: 1px solid #2d2d2d;
                border-radius: 12px;
                padding: 8px;
            }

            #PanelTitle {
                font-size: 18px;
                font-weight: bold;
                color: #ffffff;
            }

            #WarningText {
                color: #d7b56d;
                background: #1b1710;
                border: 1px solid #493b1d;
                border-radius: 8px;
                padding: 10px;
            }

            QLineEdit, QComboBox, QSpinBox, QTextEdit {
                background: #202020;
                color: #ffffff;
                border: 1px solid #444444;
                border-radius: 8px;
                padding: 7px;
            }

            QPushButton {
                background: #252525;
                color: #ffffff;
                border: 1px solid #444444;
                border-radius: 9px;
                padding: 8px 14px;
            }

            QPushButton:hover {
                background: #333333;
            }

            QPushButton:disabled {
                color: #777777;
                background: #1a1a1a;
            }

            QCheckBox {
                padding: 4px;
            }

            QProgressBar {
                background: #202020;
                border: 1px solid #444444;
                border-radius: 7px;
                height: 14px;
            }

            QProgressBar::chunk {
                background: #e8e8e8;
                border-radius: 7px;
            }
        """)

    def closeEvent(self, event) -> None:
        if self.download_thread is not None and self.download_thread.isRunning():
            QMessageBox.warning(
                self,
                "Download active",
                "Завантаження ще йде. Дочекайся завершення або пізніше додамо Cancel.",
            )
            event.ignore()
            return

        if self.wake_calibration_thread is not None and self.wake_calibration_thread.isRunning():
            QMessageBox.warning(
                self,
                "Wake calibration active",
                "Калібровка ще виконується. Дочекайся завершення.",
            )
            event.ignore()
            return

        if self.voice_test_thread is not None and self.voice_test_thread.isRunning():
            QMessageBox.warning(
                self,
                "Voice test active",
                "Тест голосу ще виконується. Дочекайся завершення.",
            )
            event.ignore()
            return

        super().closeEvent(event)