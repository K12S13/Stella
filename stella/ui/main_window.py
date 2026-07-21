from __future__ import annotations

from datetime import datetime
from typing import Any

from stella.llm.router import LLMRouter

from PySide6.QtCore import QObject, QThread, Signal
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

from stella.commands.executor import CommandExecutor, ExecutionResult
from stella.commands.matcher import CommandMatcher, MatchResult
from stella.commands.registry import CommandRegistry
from stella.core.config import load_config
from stella.core.history import HistoryLogger


class CommandWorker(QObject):
    finished = Signal(object)
    failed = Signal(str)

    def __init__(self, executor: CommandExecutor, match: MatchResult, confirmed: bool = False) -> None:
        super().__init__()
        self.executor = executor
        self.match = match
        self.confirmed = confirmed

    def run(self) -> None:
        try:
            result = self.executor.execute_match(self.match, confirmed=self.confirmed)
            self.finished.emit(result)
        except Exception as error:
            self.failed.emit(str(error))

class LLMWorker(QObject):
    finished = Signal(object)
    failed = Signal(str)

    def __init__(self, llm_router: LLMRouter, text: str) -> None:
        super().__init__()
        self.llm_router = llm_router
        self.text = text

    def run(self) -> None:
        try:
            result = self.llm_router.ask(self.text)
            self.finished.emit(result)
        except Exception as error:
            self.failed.emit(str(error))


class StellaMainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()

        self.config = load_config()
        self.registry = CommandRegistry.from_yaml()
        self.matcher = CommandMatcher(self.registry, self.config)
        self.executor = CommandExecutor(self.config)
        self.history = HistoryLogger()
        self.llm_router = LLMRouter(self.config)

        self.pending_match: MatchResult | None = None
        self.worker_thread: QThread | None = None
        self.worker: CommandWorker | None = None
        self.llm_thread: QThread | None = None
        self.llm_worker: LLMWorker | None = None

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
        self.write_system("Command execution now runs in background thread.")

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

            QPushButton:disabled {
                background: #1c1c1c;
                color: #777777;
                border: 1px solid #2a2a2a;
            }

            QLineEdit:disabled {
                background: #161616;
                color: #777777;
            }
        """)

    def start_llm_worker(self, text: str) -> None:
        self.set_busy(True)
        self.set_status(f"Thinking locally: {self.config.local_model}")

        self.llm_thread = QThread()
        self.llm_worker = LLMWorker(self.llm_router, text)

        self.llm_worker.moveToThread(self.llm_thread)

        self.llm_thread.started.connect(self.llm_worker.run)
        self.llm_worker.finished.connect(self.on_llm_finished)
        self.llm_worker.failed.connect(self.on_llm_failed)

        self.llm_worker.finished.connect(self.llm_thread.quit)
        self.llm_worker.failed.connect(self.llm_thread.quit)

        self.llm_worker.finished.connect(self.llm_worker.deleteLater)
        self.llm_worker.failed.connect(self.llm_worker.deleteLater)

        self.llm_thread.finished.connect(self.llm_thread.deleteLater)
        self.llm_thread.finished.connect(self.cleanup_llm_refs)

        self.llm_thread.start()

    def on_llm_finished(self, result: object) -> None:
        self.set_busy(False)

        if getattr(result, "success", False):
            provider = getattr(result, "provider", "ollama")
            model = getattr(result, "model", self.config.local_model)
            self.write_debug(f"LLM provider={provider}, model={model}")
            self.write_stella(getattr(result, "message", ""))
            return

        self.write_stella(getattr(result, "message", "LLM повернула невідому помилку."))

    def on_llm_failed(self, error: str) -> None:
        self.set_busy(False)
        self.write_stella(f"Помилка LLM: {error}")

    def cleanup_llm_refs(self) -> None:
        self.llm_thread = None
        self.llm_worker = None

    def set_status(self, text: str) -> None:
        self.status_label.setText(f"Status: {text}")

    def set_busy(self, busy: bool) -> None:
        self.input_line.setDisabled(busy)
        self.execute_button.setDisabled(busy)

        if busy:
            self.set_status("Working")
        else:
            self.set_status("Ready")

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

    def write_debug(self, message: str) -> None:
        self.append_log("Debug", message)

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
        self.write_debug(debug_message)

        if match.matched:
            self.start_command_worker(match, confirmed=False)
        else:
            self.write_debug("No command matched. Routing to local LLM.")
            self.start_llm_worker(text)

    def handle_confirmation(self, text: str) -> None:
        if self.pending_match is None:
            self.write_stella("Немає дії для підтвердження.")
            return

        if text in {"так", "yes", "y", "підтверджую", "confirm", "виконуй"}:
            match = self.pending_match
            self.pending_match = None
            self.start_command_worker(match, confirmed=True)
            return

        if text in {"ні", "no", "n", "скасувати", "cancel"}:
            self.pending_match = None
            self.set_status("Ready")
            self.write_stella("Скасовано.")
            return

        self.write_stella("Очікую підтвердження: так / ні")

    def start_command_worker(self, match: MatchResult, confirmed: bool = False) -> None:
        self.set_busy(True)

        self.worker_thread = QThread()
        self.worker = CommandWorker(self.executor, match, confirmed=confirmed)

        self.worker.moveToThread(self.worker_thread)

        self.worker_thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.on_command_finished)
        self.worker.failed.connect(self.on_command_failed)

        self.worker.finished.connect(self.worker_thread.quit)
        self.worker.failed.connect(self.worker_thread.quit)

        self.worker.finished.connect(self.worker.deleteLater)
        self.worker.failed.connect(self.worker.deleteLater)

        self.worker_thread.finished.connect(self.worker_thread.deleteLater)
        self.worker_thread.finished.connect(self.cleanup_worker_refs)

        self.worker_thread.start()

    def on_command_finished(self, result: ExecutionResult) -> None:
        if result.requires_confirmation:
            # Важливо: pending_match уже має бути тим самим match, який ми виконували.
            # Беремо його з worker перед cleanup.
            if self.worker is not None:
                self.pending_match = self.worker.match

            self.set_busy(False)
            self.set_status("Waiting confirmation")
            self.write_stella(result.message)
            self.write_stella("Напиши: так / ні")
            return

        self.set_busy(False)
        self.write_stella(result.message)

    def on_command_failed(self, error: str) -> None:
        self.set_busy(False)
        self.write_stella(f"Помилка виконання: {error}")

    def cleanup_worker_refs(self) -> None:
        self.worker_thread = None
        self.worker = None