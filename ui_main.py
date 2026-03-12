import os
import sys
import cv2

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QPlainTextEdit,
    QProgressBar,
    QSpinBox,
    QWidget,
)
from PySide6.QtGui import QImage, QPixmap

from Zidongkoutu import ensure_bgra, process_directory, read_image_unicode


class ProcessWorker(QThread):
    progress = Signal(int, int, str, str, bool)
    finished = Signal(int, int)

    def __init__(self, input_dir, output_dir, white_trigger, parent=None):
        super().__init__(parent)
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.white_trigger = white_trigger

    def run(self):
        def _on_progress(current, total, input_path, output_path, ok):
            self.progress.emit(current, total, input_path, output_path, ok)

        success_count, total_count = process_directory(
            self.input_dir,
            self.output_dir,
            white_trigger=self.white_trigger,
            on_progress=_on_progress,
        )
        self.finished.emit(success_count, total_count)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PictureCleaner")
        self.worker = None

        central = QWidget(self)
        layout = QGridLayout(central)

        self.input_edit = QLineEdit()
        self.output_edit = QLineEdit()
        self.input_button = QPushButton("选择输入目录")
        self.output_button = QPushButton("选择输出目录")

        self.threshold_spin = QSpinBox()
        self.threshold_spin.setRange(0, 255)
        self.threshold_spin.setValue(235)

        self.start_button = QPushButton("开始处理")
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)

        self.log_edit = QPlainTextEdit()
        self.log_edit.setReadOnly(True)

        self.preview_label = QLabel("等待预览")
        self.preview_label.setMinimumSize(320, 240)
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setStyleSheet("border: 1px solid #999;")
        self.preview_label.setSizePolicy(self.preview_label.sizePolicy().horizontalPolicy(), self.preview_label.sizePolicy().verticalPolicy())

        base_dir = os.path.dirname(
            os.path.abspath(sys.executable if getattr(sys, "frozen", False) else __file__)
        )
        default_input = os.path.join(base_dir, "Input")
        default_output = os.path.join(base_dir, "Output")
        self.input_edit.setText(default_input)
        self.output_edit.setText(default_output)

        layout.addWidget(self.preview_label, 0, 0, 1, 3)
        layout.addWidget(self.progress_bar, 1, 0, 1, 3)

        layout.addWidget(QLabel("输入目录"), 2, 0)
        layout.addWidget(self.input_edit, 2, 1)
        layout.addWidget(self.input_button, 2, 2)

        layout.addWidget(QLabel("输出目录"), 3, 0)
        layout.addWidget(self.output_edit, 3, 1)
        layout.addWidget(self.output_button, 3, 2)

        layout.addWidget(QLabel("白色阈值"), 4, 0)
        layout.addWidget(self.threshold_spin, 4, 1)

        layout.addWidget(self.log_edit, 5, 0, 1, 3)

        button_row = QHBoxLayout()
        button_row.addWidget(self.start_button)
        layout.addLayout(button_row, 6, 0, 1, 3)

        self.setCentralWidget(central)

        self.input_button.clicked.connect(self.select_input_dir)
        self.output_button.clicked.connect(self.select_output_dir)
        self.start_button.clicked.connect(self.start_processing)

    def select_input_dir(self):
        selected = QFileDialog.getExistingDirectory(self, "选择输入目录", self.input_edit.text())
        if selected:
            self.input_edit.setText(selected)

    def select_output_dir(self):
        selected = QFileDialog.getExistingDirectory(self, "选择输出目录", self.output_edit.text())
        if selected:
            self.output_edit.setText(selected)

    def start_processing(self):
        input_dir = self.input_edit.text().strip()
        output_dir = self.output_edit.text().strip()

        if not input_dir or not os.path.isdir(input_dir):
            QMessageBox.warning(self, "提示", "请输入有效的输入目录")
            return

        if not output_dir or not os.path.isdir(output_dir):
            QMessageBox.warning(self, "提示", "请输入有效的输出目录")
            return

        self.log_edit.clear()
        self.progress_bar.setValue(0)
        self.preview_label.setText("等待预览")
        self.preview_label.setPixmap(QPixmap())
        self.start_button.setEnabled(False)
        self.input_button.setEnabled(False)
        self.output_button.setEnabled(False)

        white_trigger = self.threshold_spin.value()
        self.worker = ProcessWorker(input_dir, output_dir, white_trigger, self)
        self.worker.progress.connect(self.on_progress)
        self.worker.finished.connect(self.on_finished)
        self.worker.start()

    def _to_qimage(self, img):
        if img is None:
            return None

        if len(img.shape) == 2:
            height, width = img.shape
            bytes_per_line = width
            return QImage(img.data, width, height, bytes_per_line, QImage.Format_Grayscale8)

        if img.shape[2] == 3:
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            height, width = rgb.shape[:2]
            bytes_per_line = width * 3
            return QImage(rgb.data, width, height, bytes_per_line, QImage.Format_RGB888)

        rgba = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
        height, width = rgba.shape[:2]
        bytes_per_line = width * 4
        return QImage(rgba.data, width, height, bytes_per_line, QImage.Format_RGBA8888)

    def _show_preview(self, image_path):
        img = read_image_unicode(image_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            self.preview_label.setText("无法预览")
            self.preview_label.setPixmap(QPixmap())
            return

        img = ensure_bgra(img)
        qimage = self._to_qimage(img)
        if qimage is None:
            self.preview_label.setText("无法预览")
            self.preview_label.setPixmap(QPixmap())
            return

        pixmap = QPixmap.fromImage(qimage)
        target_size = self.preview_label.size()
        scaled = pixmap.scaled(target_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.preview_label.setPixmap(scaled)
        self.preview_label.setText("")

    def on_progress(self, current, total, input_path, output_path, ok):
        if total > 0:
            percent = int(current * 100 / total)
        else:
            percent = 0
        self.progress_bar.setValue(percent)

        status = "成功" if ok else "失败"
        message = f"{current}/{total} {status}: {os.path.basename(input_path)} -> {os.path.basename(output_path)}"
        self.log_edit.appendPlainText(message)

        preview_path = output_path if ok and os.path.isfile(output_path) else input_path
        if os.path.isfile(preview_path):
            self._show_preview(preview_path)
        else:
            self.preview_label.setText("无法预览")
            self.preview_label.setPixmap(QPixmap())

    def on_finished(self, success_count, total_count):
        self.start_button.setEnabled(True)
        self.input_button.setEnabled(True)
        self.output_button.setEnabled(True)
        QMessageBox.information(self, "完成", f"处理完成：{success_count}/{total_count}")
        self.worker = None


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.resize(720, 520)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
