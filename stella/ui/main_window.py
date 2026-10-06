from __future__ import annotations

from datetime import datetime
from typing import Any

from stella.llm.router import LLMRouter
from stella.stt.stt_worker import STTWorker
from stella.stt.preload_worker import STTPreloadWorker
from stella.stt.wake_worker import WakeCommandWorker

from collections import deque
from stella.tts.tts_worker import TTSWorker
from PySide6.QtGui import QTextCursor
from PySide6.QtCore import QObject, QThread, Signal, QTimer
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPushButton,
    QTextEdit,
    QPlainTextEdit,
    QVBoxLayout,
    QWidget,
)
from stella.llm.stream_worker import LLMStreamWorker
from stella.commands.executor import CommandExecutor, ExecutionResult
from stella.commands.matcher import CommandMatcher, MatchResult
from stella.commands.registry import CommandRegistry
from stella.core.config import load_config
from stella.core.history import HistoryLogger
from stella.tts.pipeline import TTSPipeline

from stella.core.config import load_config
from stella.tts.router import TTSRouter
from stella.ui.settings_window import SettingsWindow


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

        self.tts_pipeline = TTSPipeline(self.config)
        self.tts_pipeline.start()

        self.registry = CommandRegistry.from_yaml()
        self.matcher = CommandMatcher(self.registry, self.config)
        self.executor = CommandExecutor(self.config)
        self.history = HistoryLogger()

        self.llm_router = LLMRouter(self.config)
        self.tts_router = TTSRouter(self.config)

        self.settings_window: SettingsWindow | None = None

        self.pending_match: MatchResult | None = None

        self.worker_thread: QThread | None = None
        self.worker: CommandWorker | None = None

        self.llm_thread: QThread | None = None
        self.llm_worker: LLMWorker | None = None

        self.llm_stream_thread: QThread | None = None
        self.llm_stream_worker: LLMStreamWorker | None = None
        self.current_stream_text = ""

        self.tts_thread: QThread | None = None
        self.tts_worker: TTSWorker | None = None
        self.tts_queue: deque[tuple[str, str]] = deque(maxlen=5)

        self.stt_thread: QThread | None = None
        self.stt_worker: STTWorker | None = None

        self.stt_preload_thread: QThread | None = None
        self.stt_preload_worker: STTPreloadWorker | None = None

        self.wake_thread: QThread | None = None
        self.wake_worker: WakeCommandWorker | None = None
        self.wake_restart_pending = False

        self.setWindowTitle("Stella Core v0.1")
        self.resize(900, 560)

        root = QWidget(self)
        main_layout = QVBoxLayout(root)

        self.status_label = QLabel("Status: Ready")
        self.status_label.setObjectName("statusLabel")
        main_layout.addWidget(self.status_label)

        self.log_view = QTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setPlaceholderText("Stella log...")
        main_layout.addWidget(self.log_view, 1)

        bottom_layout = QHBoxLayout()

        self.input_line = QLineEdit()
        self.input_line.setPlaceholderText("Напиши повідомлення або скажи: Стелла...")
        self.input_line.returnPressed.connect(self.execute_from_input)
        bottom_layout.addWidget(self.input_line, 1)

        self.execute_button = QPushButton("Send")
        self.execute_button.clicked.connect(self.execute_from_input)
        bottom_layout.addWidget(self.execute_button)

        self.send_button = self.execute_button

        self.clear_button = QPushButton("Clear view")
        self.clear_button.clicked.connect(self.clear_view)
        bottom_layout.addWidget(self.clear_button)

        self.settings_button = QPushButton("Settings")
        self.settings_button.clicked.connect(self.open_settings)
        bottom_layout.addWidget(self.settings_button)

        self.voice_status_label = QLabel("Wake: active")
        self.voice_status_label.setObjectName("VoiceStatusLabel")
        bottom_layout.addWidget(self.voice_status_label)

        main_layout.addLayout(bottom_layout)

        self.setCentralWidget(root)

        self.apply_style()

        self.write_system("Stella UI started.")
        self.write_system(f"Assistant name: {self.config.assistant_name}")
        self.write_system("Manual Record button removed. Wake mode is used for voice input.")

        QTimer.singleShot(500, self.start_stt_preload_worker)

        if getattr(self.config, "wake_enabled", False):
            QTimer.singleShot(1800, self.start_wake_listener)

    def start_wake_listener(self) -> None:
        if not self.config.wake_enabled:
            return

        if self.wake_thread is not None and self.wake_thread.isRunning():
            return

        if self.stt_thread is not None and self.stt_thread.isRunning():
            return

        self.write_debug("Starting wake listener...")

        self.wake_thread = QThread()
        self.wake_worker = WakeCommandWorker(self.config)

        self.wake_worker.moveToThread(self.wake_thread)

        self.wake_thread.started.connect(self.wake_worker.run)

        self.wake_worker.status.connect(self.on_wake_status)
        self.wake_worker.recognized.connect(self.on_wake_command_recognized)
        self.wake_worker.failed.connect(self.on_wake_failed)

        self.wake_worker.finished.connect(self.wake_thread.quit)
        self.wake_worker.finished.connect(self.wake_worker.deleteLater)

        self.wake_thread.finished.connect(self.wake_thread.deleteLater)
        self.wake_thread.finished.connect(self.cleanup_wake_refs)

        self.wake_thread.start()

    def stop_wake_listener(self) -> None:
        if self.wake_worker is not None:
            self.wake_worker.stop()

    def on_wake_status(self, message: str) -> None:
        self.write_debug(message)
        self.set_status(message)

    def on_wake_failed(self, message: str) -> None:
        self.write_debug(f"Wake listener failed: {message}")
        self.set_status("Ready")

        # Restart after a small delay unless app is closing.
        QTimer.singleShot(2500, self.start_wake_listener)

    def on_wake_command_recognized(self, text: str) -> None:
        recognized = text.strip()

        if not recognized:
            QTimer.singleShot(800, self.start_wake_listener)
            return

        self.write_debug(f"Wake command recognized: {recognized}")
        self.write_user(recognized)
        self.set_status("Processing voice command")

        if hasattr(self, "voice_status_label"):
            self.voice_status_label.setText("Processing voice...")

        self.wake_restart_pending = True

        QTimer.singleShot(
            0,
            lambda command=recognized: self.execute_voice_command(command),
        )

        QTimer.singleShot(25000, self.restart_wake_if_pending)

    def cleanup_wake_refs(self) -> None:
        self.wake_thread = None
        self.wake_worker = None

    def restart_wake_if_pending(self) -> None:
        if not self.wake_restart_pending:
            return

        self.wake_restart_pending = False
        self.start_wake_listener()

    def start_stt_preload_worker(self) -> None:
        if self.stt_preload_thread is not None and self.stt_preload_thread.isRunning():
            return

        if self.config.stt_provider != "whisper":
            return

        self.write_debug("Preloading Whisper STT model...")
        self.set_status("Preloading STT")

        self.stt_preload_thread = QThread()
        self.stt_preload_worker = STTPreloadWorker(self.config)

        self.stt_preload_worker.moveToThread(self.stt_preload_thread)

        self.stt_preload_thread.started.connect(self.stt_preload_worker.run)
        self.stt_preload_worker.finished.connect(self.on_stt_preload_finished)
        self.stt_preload_worker.finished.connect(self.stt_preload_thread.quit)
        self.stt_preload_worker.finished.connect(self.stt_preload_worker.deleteLater)

        self.stt_preload_thread.finished.connect(self.stt_preload_thread.deleteLater)
        self.stt_preload_thread.finished.connect(self.cleanup_stt_preload_refs)

        self.stt_preload_thread.start()

    def on_stt_preload_finished(self, ok: bool, message: str) -> None:
        if ok:
            self.write_debug(message)
        else:
            self.write_debug(message)

        self.set_status("Ready")

    def cleanup_stt_preload_refs(self) -> None:
        self.stt_preload_thread = None
        self.stt_preload_worker = None

    def start_stt_worker(self) -> None:
        # Same button behavior as ChatGPT voice input:
        # first click starts recording, second click stops recording.
        if self.stt_thread is not None and self.stt_thread.isRunning():
            self.stop_stt_worker()
            return

        self.write_debug("STT recording started.")
        self.set_status("Recording")

        if hasattr(self, "listen_button"):
            self.listen_button.setEnabled(True)
            self.listen_button.setText("■ Stop")

        self.stt_thread = QThread()
        self.stt_worker = STTWorker(self.config)

        self.stt_worker.moveToThread(self.stt_thread)

        self.stt_thread.started.connect(self.stt_worker.run)
        self.stt_worker.finished.connect(self.on_stt_finished)
        self.stt_worker.finished.connect(self.stt_thread.quit)
        self.stt_worker.finished.connect(self.stt_worker.deleteLater)

        self.stt_thread.finished.connect(self.stt_thread.deleteLater)
        self.stt_thread.finished.connect(self.cleanup_stt_refs)

        self.stt_thread.start()

    def stop_stt_worker(self) -> None:
        self.write_debug("STT recording stop requested.")
        self.set_status("Processing voice")

        if self.stt_worker is not None:
            self.stt_worker.stop()

        if hasattr(self, "listen_button"):
            self.listen_button.setEnabled(False)
            self.listen_button.setText("Processing...")

    def on_stt_finished(
        self,
        success: bool,
        text: str,
        error: str,
        provider: str,
    ) -> None:
        if hasattr(self, "listen_button"):
            self.listen_button.setEnabled(True)
            self.listen_button.setText("🎙 Listen")

        if not success:
            self.write_debug(f"STT failed [{provider}]: {error}")
            self.set_status("Ready")
            return

        recognized = text.strip()

        if not recognized:
            self.write_debug(f"STT returned empty text [{provider}].")
            self.set_status("Ready")
            return

        self.write_debug(f"STT recognized [{provider}]: {recognized}")

        self.input_line.setText(recognized)
        self.input_line.setFocus()
        self.set_status("Ready")

        if self.config.stt_auto_submit:
            QTimer.singleShot(0, self.execute_from_input)

    def cleanup_stt_refs(self) -> None:
        self.stt_thread = None
        self.stt_worker = None

    def open_settings(self) -> None:
        self.stop_wake_listener()

        if self.settings_window is None:
            self.settings_window = SettingsWindow(
                parent=self,
                on_saved=self.on_settings_saved,
            )
            self.settings_window.destroyed.connect(self.on_settings_window_destroyed)

        self.settings_window.show()
        self.settings_window.raise_()
        self.settings_window.activateWindow()

    def on_settings_saved(self) -> None:
        self.reload_runtime_config()

    def on_settings_window_destroyed(self) -> None:
        self.settings_window = None
        self.reload_runtime_config()

        if getattr(self.config, "wake_enabled", False):
            QTimer.singleShot(1000, self.start_wake_listener)

    def reload_runtime_config(self) -> None:
        self.config = load_config()
        if hasattr(self, "tts_pipeline"):
            self.tts_pipeline.stop()

        self.tts_pipeline = TTSPipeline(self.config)
        self.tts_pipeline.start()
        self.executor.config = self.config
        self.llm_router = LLMRouter(self.config)
        self.tts_router = TTSRouter(self.config)
        self.write_system("Settings reloaded.")
        QTimer.singleShot(500, self.start_stt_preload_worker)

    def test_tts(self) -> None:
        result = self.tts_router.speak("Привіт. Я Стелла. Голосовий модуль працює.")
        if result.success:
            self.write_debug("TTS test OK.")
        else:
            self.write_stella(f"TTS test failed: {result.message}")

    def speak_if_needed(self, text: str, source: str) -> None:
        text = text.strip()

        if not text:
            return

        if not self.config.tts_enabled:
            return

        if source == "llm" and not self.config.tts_speak_llm_answers:
            return

        if source == "command" and not self.config.tts_speak_command_results:
            return

        self.tts_pipeline.enqueue(text, source=source)

    def enqueue_tts(self, text: str, source: str) -> None:
        if self.tts_thread is not None and self.tts_thread.isRunning():
            self.tts_queue.append((text, source))
            self.write_debug(f"TTS queued. Queue size: {len(self.tts_queue)}")
            return

        self.start_tts_worker(text, source)

    def start_tts_worker(self, text: str, source: str) -> None:
        self.write_debug(f"TTS speaking async from source={source}...")
        self.set_status("Speaking")

        self.tts_thread = QThread()
        self.tts_worker = TTSWorker(self.config, text, source)

        self.tts_worker.moveToThread(self.tts_thread)

        self.tts_thread.started.connect(self.tts_worker.run)
        self.tts_worker.finished.connect(self.on_tts_finished)
        self.tts_worker.finished.connect(self.tts_thread.quit)
        self.tts_worker.finished.connect(self.tts_worker.deleteLater)

        self.tts_thread.finished.connect(self.tts_thread.deleteLater)
        self.tts_thread.finished.connect(self.cleanup_tts_refs)

        self.tts_thread.start()

    def on_tts_finished(self, ok: bool, message: str, source: str) -> None:
        if ok:
            self.write_debug(f"TTS OK from source={source}.")
        else:
            self.write_debug(f"TTS failed from source={source}: {message}")

        self.set_status("Ready")

    def cleanup_tts_refs(self) -> None:
        self.tts_thread = None
        self.tts_worker = None

        if self.tts_queue:
            next_text, next_source = self.tts_queue.popleft()
            QTimer.singleShot(0, lambda: self.start_tts_worker(next_text, next_source))

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
        self.lock_prompt_input()

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

    def on_llm_finished(self, result) -> None:
        message = getattr(result, "message", "").strip()

        if not message:
            message = "Ollama не повернула текст."

        self.write_stella(message)

        provider = getattr(result, "provider", "unknown")
        model = getattr(result, "model", "unknown")

        self.write_debug(f"LLM finished. Provider={provider}, model={model}")

        # Prompt must be usable immediately after text response.
        self.unlock_prompt_input()
        self.set_status("Ready")

        # TTS must never block the prompt or UI repaint.
        if self.config.tts_enabled and self.config.tts_speak_llm_answers:
            QTimer.singleShot(0, lambda text=message: self.speak_if_needed(text, source="llm"))

    def on_llm_failed(self, error: str) -> None:
        self.set_busy(False)
        self.unlock_prompt_input()
        self.set_status("Ready")
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
            self.start_llm_stream_worker(text)

    def start_llm_stream_worker(self, text: str) -> None:
        if self.llm_stream_thread is not None and self.llm_stream_thread.isRunning():
            self.write_debug("LLM stream already running.")
            return

        self.lock_prompt_input()
        self.set_status("Thinking")
        self.current_stream_text = ""

        self.begin_stella_stream_answer()

        self.llm_stream_thread = QThread()
        self.llm_stream_worker = LLMStreamWorker(self.config, text)

        self.llm_stream_worker.moveToThread(self.llm_stream_thread)

        self.llm_stream_thread.started.connect(self.llm_stream_worker.run)
        self.llm_stream_worker.chunk.connect(self.on_llm_stream_chunk)
        self.llm_stream_worker.speak_chunk.connect(self.on_llm_stream_speak_chunk)
        self.llm_stream_worker.finished.connect(self.on_llm_stream_finished)
        self.llm_stream_worker.failed.connect(self.on_llm_stream_failed)

        self.llm_stream_worker.finished.connect(self.llm_stream_thread.quit)
        self.llm_stream_worker.failed.connect(self.llm_stream_thread.quit)
        self.llm_stream_worker.finished.connect(self.llm_stream_worker.deleteLater)
        self.llm_stream_worker.failed.connect(self.llm_stream_worker.deleteLater)

        self.llm_stream_thread.finished.connect(self.llm_stream_thread.deleteLater)
        self.llm_stream_thread.finished.connect(self.cleanup_llm_stream_refs)

        self.llm_stream_thread.start()

    def on_llm_stream_speak_chunk(self, text: str) -> None:
        text = text.strip()

        if not text:
            return

        self.speak_if_needed(text, source="llm")

    def on_llm_stream_chunk(self, chunk: str) -> None:
        self.current_stream_text += chunk
        self.append_stella_stream_chunk(chunk)

    def on_llm_stream_finished(
        self,
        full_text: str,
        provider: str,
        model: str,
    ) -> None:
        self.finish_stella_stream_answer()

        self.write_debug(
            f"LLM stream finished. Provider={provider}, model={model}"
        )

        self.unlock_prompt_input()
        self.set_status("Ready")
        if self.wake_restart_pending:
            self.wake_restart_pending = False

            if hasattr(self, "voice_status_label"):
                self.voice_status_label.setText("Wake mode active")

            QTimer.singleShot(1500, self.start_wake_listener)

    def on_llm_stream_failed(self, error: str) -> None:
        self.finish_stella_stream_answer()
        self.write_stella(f"LLM stream failed: {error}")

        self.unlock_prompt_input()
        self.set_status("Ready")
        if self.wake_restart_pending:
            self.wake_restart_pending = False

            if hasattr(self, "voice_status_label"):
                self.voice_status_label.setText("Wake mode active")

            QTimer.singleShot(1500, self.start_wake_listener)

    def cleanup_llm_stream_refs(self) -> None:
        self.llm_stream_thread = None
        self.llm_stream_worker = None

    def _find_text_log_widget(self):
        possible_names = [
            "log",
            "log_box",
            "chat_log",
            "output_box",
            "output_text",
            "conversation_view",
            "history_view",
        ]

        for name in possible_names:
            widget = getattr(self, name, None)

            if isinstance(widget, (QTextEdit, QPlainTextEdit)):
                return widget

        text_edits = self.findChildren(QTextEdit)

        if text_edits:
            return text_edits[0]

        plain_text_edits = self.findChildren(QPlainTextEdit)

        if plain_text_edits:
            return plain_text_edits[0]

        return None

    def begin_stella_stream_answer(self) -> None:
        widget = self._find_text_log_widget()

        if widget is None:
            return

        timestamp = datetime.now().strftime("%H:%M:%S")
        widget.moveCursor(QTextCursor.MoveOperation.End)

        # One assistant message block. Chunks will be appended into this same block.
        widget.insertPlainText(f"\n[{timestamp}] Stella: ")
        widget.ensureCursorVisible()

    def append_stella_stream_chunk(self, chunk: str) -> None:
        widget = self._find_text_log_widget()

        if widget is None:
            return

        widget.moveCursor(QTextCursor.MoveOperation.End)
        widget.insertPlainText(chunk)
        widget.ensureCursorVisible()

    def finish_stella_stream_answer(self) -> None:
        widget = self._find_text_log_widget()

        if widget is None:
            return

        widget.moveCursor(QTextCursor.MoveOperation.End)

        # Only close the visual block after the whole LLM stream is finished.
        widget.insertPlainText("\n")
        widget.ensureCursorVisible()

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
            self.speak_if_needed(result.message, source="command")
            self.write_stella("Напиши: так / ні")
            return

        self.set_busy(False)
        self.write_stella(result.message)
        if self.wake_restart_pending:
            self.wake_restart_pending = False

            if hasattr(self, "voice_status_label"):
                self.voice_status_label.setText("Wake mode active")

            QTimer.singleShot(1000, self.start_wake_listener)

    def on_command_failed(self, error: str) -> None:
        self.set_busy(False)
        self.write_stella(f"Помилка виконання: {error}")

    def cleanup_worker_refs(self) -> None:
        self.worker_thread = None
        self.worker = None

    def _send_button(self):
        return getattr(self, "send_button", getattr(self, "execute_button", None))

    def lock_prompt_input(self) -> None:
        if hasattr(self, "input_line"):
            self.input_line.setEnabled(False)

        send_button = self._send_button()

        if send_button is not None:
            send_button.setEnabled(False)

    def unlock_prompt_input(self) -> None:
        if hasattr(self, "input_line"):
            self.input_line.setEnabled(True)
            self.input_line.setFocus()

        send_button = self._send_button()

        if send_button is not None:
            send_button.setEnabled(True)

    def execute_voice_command(self, text: str) -> None:
        clean_text = text.strip()

        if not clean_text:
            QTimer.singleShot(800, self.start_wake_listener)
            return

        self.input_line.setText(clean_text)
        self.execute_from_input()

    def closeEvent(self, event) -> None:
        active_threads = []
        if self.llm_stream_thread is not None and self.llm_stream_thread.isRunning():
            active_threads.append("LLM Stream")

        if self.stt_thread is not None and self.stt_thread.isRunning():
            active_threads.append("STT")

        if self.tts_thread is not None and self.tts_thread.isRunning():
            active_threads.append("TTS")

        if self.llm_thread is not None and self.llm_thread.isRunning():
            active_threads.append("LLM")

        if self.worker_thread is not None and self.worker_thread.isRunning():
            active_threads.append("Command")

        if self.stt_preload_thread is not None and self.stt_preload_thread.isRunning():
            active_threads.append("STT Preload")

        if active_threads:
            self.write_debug(
                "Background task still running: "
                + ", ".join(active_threads)
                + ". Hiding window instead of killing thread."
            )
            self.hide()
            event.ignore()
            return
        if hasattr(self, "tts_pipeline"):
            self.tts_pipeline.stop()
        super().closeEvent(event)