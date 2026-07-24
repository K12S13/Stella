from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QObject, Qt, QThread, Signal
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


class SettingsWindow(QWidget):
    def __init__(self, on_saved=None) -> None:
        super().__init__()

        self.on_saved = on_saved
        self.store = SettingsStore()
        self.data = self.store.load_raw()
        self.hardware = detect_hardware()
        self.ollama = OllamaManager()

        self.download_thread: QThread | None = None
        self.download_worker: DownloadWorker | None = None

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

    def create_fields(self) -> None:
        self.assistant_name = QLineEdit(self._get("assistant.name", "стелла"))

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
        assistant_group.setLayout(assistant_form)

        layout.addWidget(budget_group)
        layout.addWidget(assistant_group)
        layout.addStretch()

        page.setLayout(layout)
        return page

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

        layout.addWidget(self.page_title("Voice / TTS"))

        tts_group = QGroupBox("TTS behavior")
        tts_form = QFormLayout()
        tts_form.addRow("", self.tts_enabled)
        tts_form.addRow("", self.speak_llm)
        tts_form.addRow("", self.speak_commands)
        tts_group.setLayout(tts_form)

        voice_group = QGroupBox("Voice catalog")
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

        voice_form = QFormLayout()
        voice_form.addRow("Piper model path:", self.tts_model)
        voice_form.addRow("Output player:", self.output_player)

        voice_layout.addLayout(voice_form)
        voice_group.setLayout(voice_layout)

        layout.addWidget(tts_group)
        layout.addWidget(voice_group)
        layout.addStretch()

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

        test_llm = QPushButton("Test LLM")
        test_llm.clicked.connect(lambda: self.status_label.setText("Test LLM from main window for now."))

        test_tts = QPushButton("Test TTS")
        test_tts.clicked.connect(lambda: self.status_label.setText("Use Test TTS in main window for now."))

        layout.addWidget(title)
        layout.addWidget(self.summary_label)
        layout.addStretch()
        layout.addWidget(test_llm)
        layout.addWidget(test_tts)

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
        self.store.set_value("assistant.name", self.assistant_name.text().strip() or "стелла")
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

        self.summary_label.setText(
            f"Mode: {self.resource_mode.currentText()}\n\n"
            f"Model: {self.local_model.text().strip() or 'none'}\n"
            f"Context: {self.num_ctx.value()}\n\n"
            f"RAM budget: {self.ram_budget.value()} GB\n"
            f"VRAM budget: {self.vram_budget.value()} GB\n"
            f"CPU threads: {self.cpu_threads.value()}\n\n"
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