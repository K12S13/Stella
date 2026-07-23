from __future__ import annotations

from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from stella.core.settings_store import SettingsStore


class SettingsWindow(QWidget):
    def __init__(self, on_saved=None) -> None:
        super().__init__()

        self.on_saved = on_saved
        self.store = SettingsStore()
        self.data = self.store.load_raw()

        self.setWindowTitle("Stella Settings")
        self.resize(520, 360)

        self.assistant_name = QLineEdit(self._get("assistant.name", "стелла"))

        self.resource_mode = QComboBox()
        self.resource_mode.addItems(["eco", "balanced", "strong", "offline"])
        self.resource_mode.setCurrentText(self._get("modes.resource_mode", "balanced"))

        self.local_model = QLineEdit(self._get("llm.local_model", "qwen2.5:3b"))

        self.num_ctx = QSpinBox()
        self.num_ctx.setRange(512, 8192)
        self.num_ctx.setSingleStep(512)
        self.num_ctx.setValue(int(self._get("llm.num_ctx", 2048)))

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

        self.save_button = QPushButton("Save settings")
        self.save_button.clicked.connect(self.save_settings)

        form = QFormLayout()
        form.addRow("Assistant name:", self.assistant_name)
        form.addRow("Resource mode:", self.resource_mode)
        form.addRow("Ollama model:", self.local_model)
        form.addRow("Ollama num_ctx:", self.num_ctx)
        form.addRow("", self.tts_enabled)
        form.addRow("", self.speak_llm)
        form.addRow("", self.speak_commands)
        form.addRow("Piper model:", self.tts_model)
        form.addRow("Output player:", self.output_player)

        layout = QVBoxLayout()
        layout.addLayout(form)
        layout.addWidget(self.save_button)

        self.setLayout(layout)
        self.apply_style()

    def _get(self, path: str, default):
        current = self.data

        for part in path.split("."):
            if not isinstance(current, dict):
                return default

            current = current.get(part)

            if current is None:
                return default

        return current

    def save_settings(self) -> None:
        self.store.set_value("assistant.name", self.assistant_name.text().strip() or "стелла")
        self.store.set_value("modes.resource_mode", self.resource_mode.currentText())
        self.store.set_value("llm.local_model", self.local_model.text().strip() or "qwen2.5:3b")
        self.store.set_value("llm.num_ctx", int(self.num_ctx.value()))

        self.store.set_value("tts.enabled", self.tts_enabled.isChecked())
        self.store.set_value("tts.provider", "piper")
        self.store.set_value("tts.speak_llm_answers", self.speak_llm.isChecked())
        self.store.set_value("tts.speak_command_results", self.speak_commands.isChecked())
        self.store.set_value("tts.piper_model", self.tts_model.text().strip())
        self.store.set_value("tts.output_player", self.output_player.text().strip() or "pw-play")

        QMessageBox.information(
            self,
            "Settings saved",
            "Налаштування збережено. Перезапусти Stella UI, щоб усе точно оновилось.",
        )

        if self.on_saved:
            self.on_saved()

    def apply_style(self) -> None:
        self.setStyleSheet("""
            QWidget {
                background: #111111;
                color: #e6e6e6;
                font-size: 14px;
            }

            QLineEdit, QComboBox, QSpinBox {
                background: #202020;
                color: #ffffff;
                border: 1px solid #444444;
                border-radius: 6px;
                padding: 6px;
            }

            QPushButton {
                background: #2d2d2d;
                color: #ffffff;
                border: 1px solid #444444;
                border-radius: 8px;
                padding: 8px 14px;
            }

            QPushButton:hover {
                background: #3a3a3a;
            }

            QCheckBox {
                padding: 4px;
            }
        """)