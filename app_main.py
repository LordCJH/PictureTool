import sys

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from ui_main import MainWindow


def main():
    app = QApplication(sys.argv)
    app.setAttribute(Qt.AA_Use96Dpi, True)
    window = MainWindow()
    window.resize(1200, 800)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
