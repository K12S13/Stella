from __future__ import annotations

import sys

from PySide6.QtWidgets import QApplication

from stella.ui.main_window import StellaMainWindow


def main() -> None:
    app = QApplication(sys.argv)

    window = StellaMainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()