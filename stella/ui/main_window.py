from __future__ import annotations

from datetime import datetime

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from stella.commands.executor import CommandExecutor
from stella.commands.matcher import CommandMatcher
from stella.commands.registry import CommandRegistry
from stella.core.config import load_config
from stella.core.history import HistoryLogger


class StellaMainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()

        self.config = load_config()
        self.registry = CommandRegistry.from_yaml()
        self.matcher = CommandMatcher(self.registry, self.config)
        self.executor = CommandExecutor(self.config)
        self.history = HistoryLogger()

        self.pending_match = None

        self.setWindowTitle("Stella Core v0.1")
        self.resize(720, 460)

        self.status_label = QLabel("Status: Ready")
        self.status_label.setObjectName("statusLabel")

        self.log_view = QTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setPlaceholderText("Stella log...")

        self.input_line = QLineEdit()
        self.input_line.setPlaceholderText("Напиши команду: стелла відкрий firefox")
        self.input_line.returnPressed.connect(self.execute_from_input)

        self.execute_button = QPushButton("Execute")
        self.execute_button.clicked.connect(self.execute_from_input)

        self.clear_button = QPushButton("Clear view")
        self.clear_button.clicked.connect(self.clear_view)

        input_layout = QHBoxLayout()
        input_layout.addWidget(self.input_line)
        input_layout.addWidget(self.execute_button)
        input_layout.addWidget(self.clear_button)

        layout = QVBoxLayout()
        layout.addWidget(self.status_label)
        layout.addWidget(self.log_view)
        layout.addLayout(input_layout)

        root = QWidget()
        root.setLayout(layout)
        self.setCentralWidget(root)

        self.apply_style()

        self.write_system("Stella UI started.")
        self.write_system(f"Assistant name: {self.config.assistant_name}")
        self.write_system("Terminal backup: python -m stella.main")

    def apply_style(self) -> None:
        self.setStyleSheet("""
            QMainWindow {
                background: #111111;
            }

            QLabel#statusLabel {
                color: #d4d4d4;
                font-size: 14px;
                padding: 6px;
            }

            QTextEdit {
                background: #181818;
                color: #e6e6e6;
                border: 1px solid #333333;
                border-radius: 8px;
                padding: 8px;
                font-size: 14px;
            }

            QLineEdit {
                background: #202020;
                color: #ffffff;
                border: 1px solid #444444;
                border-radius: 8px;
                padding: 8px;
                font-size: 14px;
            }

            QPushButton {
                background: #2d2d2d;
                color: #ffffff;
                border: 1px solid #444444;
                border-radius: 8px;
                padding: 8px 14px;
                font-size: 14px;
            }

            QPushButton:hover {
                background: #3a3a3a;
            }

            QPushButton:pressed {
                background: #4a4a4a;
            }
        """)

    def set_status(self, text: str) -> None:
        self.status_label.setText(f"Status: {text}")

    def timestamp(self) -> str:
        return datetime.now().strftime("%H:%M:%S")

    def append_log(self, role: str, message: str) -> None:
        line = f"[{self.timestamp()}] {role}: {message}"
        self.log_view.append(line)
        self.history.write(role, message)

    def write_system(self, message: str) -> None:
        self.append_log("System", message)

    def write_stella(self, message: str) -> None:
        self.append_log("Stella", message)

    def write_user(self, message: str) -> None:
        self.append_log("User", message)

    def clear_view(self) -> None:
        self.log_view.clear()
        self.write_system("View cleared. History file is still saved.")

    def execute_from_input(self) -> None:
        text = self.input_line.text().strip()

        if not text:
            return

        self.input_line.clear()
        self.write_user(text)

        lowered = text.lower().strip()

        if lowered in {"exit", "quit", "вихід", "вийти"}:
            self.close()
            return

        if self.pending_match is not None:
            self.handle_confirmation(lowered)
            return

        self.set_status("Matching command")

        match = self.matcher.match(text)

        debug_message = (
            f"source={match.source}, "
            f"confidence={match.confidence:.2f}, "
            f"normalized='{match.normalized_text}'"
        )
        self.append_log("Debug", debug_message)

        self.set_status("Executing")
        result = self.executor.execute_match(match)

        if result.requires_confirmation:
            self.pending_match = match
            self.set_status("Waiting confirmation")
            self.write_stella(result.message)
            self.write_stella("Напиши: так / ні")
            return

        self.set_status("Ready")
        self.write_stella(result.message)

    def handle_confirmation(self, text: str) -> None:
        if text in {"так", "yes", "y", "підтверджую", "confirm", "виконуй"}:
            self.set_status("Executing confirmed action")
            result = self.executor.execute_match(self.pending_match, confirmed=True)
            self.pending_match = None
            self.set_status("Ready")
            self.write_stella(result.message)
            return

        if text in {"ні", "no", "n", "скасувати", "cancel"}:
            self.pending_match = None
            self.set_status("Ready")
            self.write_stella("Скасовано.")
            return

        self.write_stella("Очікую підтвердження: так / ні")