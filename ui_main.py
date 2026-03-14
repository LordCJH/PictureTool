import os
import sys
import cv2

from PySide6.QtCore import Qt, QThread, Signal, QPoint, QRect
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
    QSizePolicy,
    QSpinBox,
    QWidget,
    QDialog,
    QVBoxLayout,
    QCheckBox,
)
from PySide6.QtGui import QImage, QPixmap, QPainter, QBrush, QColor, QPalette

from Zidongkoutu import ensure_bgra, process_directory, read_image_unicode
class ClickableLabel(QLabel):
    clicked = Signal(QPoint, Qt.MouseButton)

    def mousePressEvent(self, event):
        self.clicked.emit(event.position().toPoint(), event.button())
        super().mousePressEvent(event)
class LogDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("处理日志")
        self.resize(720, 420)

        layout = QVBoxLayout(self)
        self.log_edit = QPlainTextEdit()
        self.log_edit.setReadOnly(True)
        layout.addWidget(self.log_edit)

    def append(self, text):
        self.log_edit.appendPlainText(text)


class ProcessWorker(QThread):
    progress = Signal(int, int, str, str, bool)
    finished = Signal(int, int)
    log = Signal(str)

    def __init__(self, input_dir, output_dir, white_trigger, selected_points_map, color_tolerance, do_remove_white, do_remove_points, do_resize, target_width, target_height, keep_original_name=False, parent=None):
        super().__init__(parent)
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.white_trigger = white_trigger
        self.selected_points_map = selected_points_map
        self.color_tolerance = color_tolerance
        self.do_remove_white = do_remove_white
        self.do_remove_points = do_remove_points
        self.do_resize = do_resize
        self.target_width = target_width
        self.target_height = target_height
        self.keep_original_name = keep_original_name

    def run(self):
        def _on_progress(current, total, input_path, output_path, ok):
            self.progress.emit(current, total, input_path, output_path, ok)

        try:
            self.log.emit("开始处理目录...")
            success_count, total_count = process_directory(
                self.input_dir,
                self.output_dir,
                white_trigger=self.white_trigger,
                selected_points_map=self.selected_points_map,
                color_tolerance=self.color_tolerance,
                do_remove_white=self.do_remove_white,
                do_remove_points=self.do_remove_points,
                do_resize=self.do_resize,
                target_width=self.target_width,
                target_height=self.target_height,
                keep_original_name=self.keep_original_name,
                on_progress=_on_progress,
            )
            self.log.emit("目录处理结束")
            self.finished.emit(success_count, total_count)
        except Exception as exc:
            self.log.emit(f"处理异常: {exc}")
            self.finished.emit(0, 0)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PictureCleaner")
        self.worker = None
        self.log_dialog = None
        self.selected_points_map = {}
        self.current_input_path = None
        self.current_input_key = None
        self.current_image_size = None
        self.preview_original_pixmap = None
        self.preview_display_rect = None

        central = QWidget(self)
        layout = QGridLayout(central)
        self.main_layout = layout

        self.input_edit = QLineEdit()
        self.output_edit = QLineEdit()
        self.input_button = QPushButton("选择输入目录")
        self.output_button = QPushButton("选择输出目录")
        self.image_button = QPushButton("选择图片")
        self.image_button.setEnabled(True)

        self.threshold_spin = QSpinBox()
        self.threshold_spin.setRange(0, 255)
        self.threshold_spin.setValue(235)

        self.tolerance_spin = QSpinBox()
        self.tolerance_spin.setRange(0, 255)
        self.tolerance_spin.setValue(5)

        self.start_button = QPushButton("开始处理")
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)

        self.clear_points_button = QPushButton("清除选点")

        self.remove_white_checkbox = QCheckBox("去除白边")
        self.remove_white_checkbox.setChecked(True)
        self.remove_points_checkbox = QCheckBox("选点删除")
        self.remove_points_checkbox.setChecked(True)
        self.resize_checkbox = QCheckBox("调整分辨率")
        self.resize_checkbox.setChecked(False)
        self.keep_name_checkbox = QCheckBox("保持原文件名")
        self.keep_name_checkbox.setChecked(False)

        self.resize_width_spin = QSpinBox()
        self.resize_width_spin.setRange(1, 100000)
        self.resize_width_spin.setValue(1)
        self.resize_width_spin.setEnabled(False)
        self.resize_height_spin = QSpinBox()
        self.resize_height_spin.setRange(1, 100000)
        self.resize_height_spin.setValue(1)
        self.resize_height_spin.setEnabled(False)

        self.resize_checkbox.toggled.connect(self.on_resize_toggled)

        self.preview_label = ClickableLabel("等待预览")
        self.preview_label.setMinimumSize(320, 240)
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setStyleSheet("border: 1px solid #999; background-color: #8b5a2b;")
        self.preview_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.preview_label.setAttribute(Qt.WA_TransparentForMouseEvents, False)

        base_dir = os.path.dirname(
            os.path.abspath(sys.executable if getattr(sys, "frozen", False) else __file__)
        )
        default_input = os.path.join(base_dir, "Input")
        default_output = os.path.join(base_dir, "Output")
        self.input_edit.setText(default_input)
        self.output_edit.setText(default_output)

        layout.addWidget(self.preview_label, 0, 0, 1, 4)
        layout.addWidget(self.progress_bar, 1, 0, 1, 4)

        layout.setRowStretch(0, 1)
        layout.setRowStretch(5, 0)
        layout.setColumnStretch(1, 1)
        layout.setColumnStretch(0, 0)
        layout.setColumnStretch(2, 0)
        layout.setColumnStretch(3, 0)

        layout.addWidget(QLabel("输入目录"), 2, 0)
        layout.addWidget(self.input_edit, 2, 1)
        layout.addWidget(self.input_button, 2, 2)

        point_row = QHBoxLayout()
        point_row.addWidget(self.image_button)
        point_row.addWidget(self.clear_points_button)
        layout.addLayout(point_row, 2, 3)

        layout.addWidget(QLabel("输出目录"), 3, 0)
        layout.addWidget(self.output_edit, 3, 1, 1, 2)
        layout.addWidget(self.output_button, 3, 3)

        layout.addWidget(QLabel("白色阈值"), 4, 0)
        layout.addWidget(self.threshold_spin, 4, 1)

        layout.addWidget(QLabel("选点容差"), 4, 2)
        layout.addWidget(self.tolerance_spin, 4, 3)

        option_row = QHBoxLayout()
        option_row.addWidget(self.remove_white_checkbox)
        option_row.addWidget(self.remove_points_checkbox)
        option_row.addWidget(self.resize_checkbox)
        option_row.addWidget(self.keep_name_checkbox)
        layout.addLayout(option_row, 5, 0, 1, 4)

        resize_row = QHBoxLayout()
        resize_row.addWidget(QLabel("宽"))
        resize_row.addWidget(self.resize_width_spin)
        resize_row.addWidget(QLabel("高"))
        resize_row.addWidget(self.resize_height_spin)
        resize_row.addStretch(1)
        layout.addLayout(resize_row, 6, 0, 1, 4)

        button_row = QHBoxLayout()
        button_row.addWidget(self.start_button)
        layout.addLayout(button_row, 7, 0, 1, 4)

        self.setCentralWidget(central)

        self.input_button.clicked.connect(self.select_input_dir)
        self.output_button.clicked.connect(self.select_output_dir)
        self.image_button.clicked.connect(self.select_preview_image)
        self.start_button.clicked.connect(self.start_processing)
        self.clear_points_button.clicked.connect(self.clear_selected_points)
        self.preview_label.clicked.connect(self.on_preview_clicked)

        # 程序启动时自动加载默认输入目录的第一张图片
        default_input_dir = self.input_edit.text().strip()
        if default_input_dir and os.path.isdir(default_input_dir):
            first_image = self._get_first_image_in_directory(default_input_dir)
            if first_image:
                self._show_preview(first_image)

    def on_resize_toggled(self, checked):
        self.resize_width_spin.setEnabled(checked)
        self.resize_height_spin.setEnabled(checked)
        if not checked:
            return
        if not self.current_image_size:
            return
        if self.resize_width_spin.value() <= 1:
            self.resize_width_spin.setValue(self.current_image_size[0])
        if self.resize_height_spin.value() <= 1:
            self.resize_height_spin.setValue(self.current_image_size[1])

    def clear_selected_points(self):
        if not self.current_input_key:
            return
        points = self.selected_points_map.get(self.current_input_key)
        if not points:
            return
        points.clear()
        self._refresh_preview_scaled()

    def _current_selected_points(self):
        if not self.current_input_key:
            return []
        return self.selected_points_map.setdefault(self.current_input_key, [])

    def _remove_nearest_point(self, points, x, y, radius=8):
        if not points:
            return
        radius_sq = radius * radius
        nearest_index = None
        nearest_dist = None
        for index, (px, py) in enumerate(points):
            dx = px - x
            dy = py - y
            dist_sq = dx * dx + dy * dy
            if dist_sq <= radius_sq and (nearest_dist is None or dist_sq < nearest_dist):
                nearest_dist = dist_sq
                nearest_index = index
        if nearest_index is not None:
            points.pop(nearest_index)

    def on_preview_clicked(self, position, button):
        if not self.preview_display_rect or not self.current_image_size:
            return
        if not self.preview_display_rect.contains(position):
            return

        img_w, img_h = self.current_image_size
        rect = self.preview_display_rect
        if rect.width() <= 0 or rect.height() <= 0:
            return

        x = (position.x() - rect.x()) * img_w / rect.width()
        y = (position.y() - rect.y()) * img_h / rect.height()
        img_x = int(round(x))
        img_y = int(round(y))

        if img_x < 0 or img_y < 0 or img_x >= img_w or img_y >= img_h:
            return

        points = self._current_selected_points()
        if button == Qt.RightButton:
            self._remove_nearest_point(points, img_x, img_y, radius=8)
        else:
            points.append((img_x, img_y))
        self._refresh_preview_scaled()

    def _get_first_image_in_directory(self, dir_path):
        """扫描目录下的图片文件，返回第一张图片的完整路径，如果没有则返回 None。"""
        if not dir_path or not os.path.isdir(dir_path):
            return None
        image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp')
        try:
            for name in os.listdir(dir_path):
                if name.lower().endswith(image_extensions):
                    full_path = os.path.join(dir_path, name)
                    if os.path.isfile(full_path):
                        return full_path
        except Exception:
            pass
        return None

    def select_input_dir(self):
        selected = QFileDialog.getExistingDirectory(self, "选择输入目录", self.input_edit.text())
        if selected:
            self.input_edit.setText(selected)
            # 自动加载目录下的第一张图片作为预览
            first_image = self._get_first_image_in_directory(selected)
            if first_image:
                self._show_preview(first_image)

    def select_output_dir(self):
        selected = QFileDialog.getExistingDirectory(self, "选择输出目录", self.output_edit.text())
        if selected:
            self.output_edit.setText(selected)

    def select_preview_image(self):
        start_dir = self.input_edit.text().strip() or os.getcwd()
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择图片",
            start_dir,
            "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp)",
        )
        if not file_path:
            return
        if not os.path.isfile(file_path):
            return
        self._show_preview(file_path)

    def start_processing(self):
        input_dir = self.input_edit.text().strip()
        output_dir = self.output_edit.text().strip()

        if self.current_input_path and input_dir and not os.path.commonpath([self.current_input_path, input_dir]) == input_dir:
            if self.log_dialog:
                self.log_dialog.append("当前预览图片不在输入目录内，将按文件名匹配输入目录图片")

        if not input_dir or not os.path.isdir(input_dir):
            if self.log_dialog:
                self.log_dialog.append("请输入有效的输入目录")
            return

        if not output_dir or not os.path.isdir(output_dir):
            if self.log_dialog:
                self.log_dialog.append("请输入有效的输出目录")
            return

        self.progress_bar.setValue(0)
        self.preview_original_pixmap = None
        self.current_image_size = None
        self.current_input_path = None
        self.current_input_key = None
        self.preview_display_rect = None
        self.preview_label.setText("等待预览")
        self.preview_label.setPixmap(QPixmap())
        self.start_button.setEnabled(False)
        self.input_button.setEnabled(False)
        self.output_button.setEnabled(False)
        self.image_button.setEnabled(False)
        self.clear_points_button.setEnabled(False)

        if self.log_dialog is None:
            self.log_dialog = LogDialog(self)
        self.log_dialog.log_edit.clear()
        self.log_dialog.append("开始处理...")
        self.log_dialog.show()
        self.log_dialog.raise_()

        white_trigger = self.threshold_spin.value()
        color_tolerance = self.tolerance_spin.value()
        selected_points_snapshot = self._resolve_selected_points(input_dir)
        if not selected_points_snapshot:
            self.log_dialog.append("提示: 当前没有选点，未传入选点清理")

        self.worker = ProcessWorker(
            input_dir,
            output_dir,
            white_trigger,
            selected_points_snapshot,
            color_tolerance,
            self.remove_white_checkbox.isChecked(),
            self.remove_points_checkbox.isChecked(),
            self.resize_checkbox.isChecked(),
            self.resize_width_spin.value(),
            self.resize_height_spin.value(),
            self.keep_name_checkbox.isChecked(),
            self,
        )
        self.worker.progress.connect(self.on_progress)
        self.worker.finished.connect(self.on_finished)
        self.worker.log.connect(self.on_worker_log)
        self.worker.start()

    def _resolve_selected_points(self, input_dir):
        selected_points = {}
        for path, points in self.selected_points_map.items():
            if not points:
                continue
            if path in selected_points:
                selected_points[path] = list(points)
                continue
            base_name = os.path.basename(path)
            if input_dir:
                candidate = os.path.join(input_dir, base_name)
                if os.path.isfile(candidate):
                    selected_points[candidate] = list(points)
                    continue
            selected_points[path] = list(points)
        return selected_points

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

    def _refresh_preview_scaled(self):
        if not self.preview_original_pixmap or self.preview_original_pixmap.isNull():
            return

        target_size = self.preview_label.size()
        if target_size.width() <= 0 or target_size.height() <= 0:
            return

        base_pixmap = QPixmap(self.preview_original_pixmap)
        points = self._current_selected_points()
        if points:
            painter = QPainter(base_pixmap)
            painter.setRenderHint(QPainter.Antialiasing, True)
            painter.setBrush(QBrush(QColor(220, 50, 50)))
            painter.setPen(QColor(220, 50, 50))
            for x, y in points:
                painter.drawEllipse(QPoint(x, y), 4, 4)
            painter.end()

        scaled = base_pixmap.scaled(target_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        scaled_w = scaled.width()
        scaled_h = scaled.height()
        x_offset = max(0, (target_size.width() - scaled_w) // 2)
        y_offset = max(0, (target_size.height() - scaled_h) // 2)
        self.preview_display_rect = QRect(x_offset, y_offset, scaled_w, scaled_h)

        self.preview_label.setPixmap(scaled)
        self.preview_label.setText("")

    def _show_preview(self, image_path, update_key=True):
        img = read_image_unicode(image_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            self.preview_original_pixmap = None
            self.current_image_size = None
            self.current_input_path = None
            self.current_input_key = None
            self.preview_display_rect = None
            self.preview_label.setText("无法预览")
            self.preview_label.setPixmap(QPixmap())
            return

        img = ensure_bgra(img)
        qimage = self._to_qimage(img)
        if qimage is None:
            self.preview_original_pixmap = None
            self.current_image_size = None
            self.current_input_path = None
            self.current_input_key = None
            self.preview_display_rect = None
            self.preview_label.setText("无法预览")
            self.preview_label.setPixmap(QPixmap())
            return

        self.preview_original_pixmap = QPixmap.fromImage(qimage)
        self.current_image_size = (qimage.width(), qimage.height())
        # 自动更新分辨率输入框为当前图片尺寸
        self.resize_width_spin.setValue(self.current_image_size[0])
        self.resize_height_spin.setValue(self.current_image_size[1])
        if update_key:
            self.current_input_path = image_path
            input_dir = self.input_edit.text().strip()
            current_key = image_path
            if input_dir and os.path.isdir(input_dir):
                candidate = os.path.join(input_dir, os.path.basename(image_path))
                if os.path.isfile(candidate):
                    current_key = candidate
            self.current_input_key = current_key
            if self.current_input_key not in self.selected_points_map:
                self.selected_points_map[self.current_input_key] = []
        self._refresh_preview_scaled()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._refresh_preview_scaled()

    def on_worker_log(self, message):
        if self.log_dialog:
            self.log_dialog.append(message)

    def on_progress(self, current, total, input_path, output_path, ok):
        if total > 0:
            percent = int(current * 100 / total)
        else:
            percent = 0
        self.progress_bar.setValue(percent)

        status = "成功" if ok else "失败"
        message = f"{current}/{total} {status}: {os.path.basename(input_path)} -> {os.path.basename(output_path)}"
        if self.log_dialog:
            self.log_dialog.append(message)

        preview_path = input_path
        update_key = True
        if ok:
            output_candidate = None
            if output_path and os.path.isfile(output_path):
                output_candidate = output_path
            else:
                output_dir = self.output_edit.text().strip()
                if output_dir and os.path.isdir(output_dir):
                    files = [
                        os.path.join(output_dir, name)
                        for name in os.listdir(output_dir)
                    ]
                    files = [path for path in files if os.path.isfile(path)]
                    if files:
                        output_candidate = max(files, key=os.path.getmtime)
            if output_candidate:
                preview_path = output_candidate
                update_key = False
        if os.path.isfile(preview_path):
            self._show_preview(preview_path, update_key=update_key)
        else:
            self.preview_original_pixmap = None
            self.current_image_size = None
            self.current_input_path = None
            self.current_input_key = None
            self.preview_display_rect = None
            self.preview_label.setText("无法预览")
            self.preview_label.setPixmap(QPixmap())

    def on_finished(self, success_count, total_count):
        self.start_button.setEnabled(True)
        self.input_button.setEnabled(True)
        self.output_button.setEnabled(True)
        self.image_button.setEnabled(True)
        self.clear_points_button.setEnabled(True)
        self.worker = None
        if self.log_dialog:
            self.log_dialog.append(f"处理完成：{success_count}/{total_count}")


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
