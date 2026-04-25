import os
import sys
import cv2

from PySide6.QtCore import Qt, QThread, Signal, QPoint, QRect
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
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
from PySide6.QtGui import QImage, QPixmap, QPainter, QBrush, QColor, QPalette, QPen

from Zidongkoutu import (
    ensure_bgra,
    process_directory,
    read_image_unicode,
    extract_video_frames,
    batch_rename_images,
    slice_directory_objects,
)


class ClickableLabel(QLabel):
    clicked = Signal(QPoint, Qt.MouseButton)
    pressed = Signal(QPoint, Qt.MouseButton)
    moved = Signal(QPoint, Qt.MouseButton)
    released = Signal(QPoint, Qt.MouseButton)

    def mousePressEvent(self, event):
        position = event.position().toPoint()
        self.clicked.emit(position, event.button())
        self.pressed.emit(position, event.button())
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        buttons = event.buttons()
        primary_button = Qt.LeftButton if buttons & Qt.LeftButton else Qt.NoButton
        self.moved.emit(event.position().toPoint(), primary_button)
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        self.released.emit(event.position().toPoint(), event.button())
        super().mouseReleaseEvent(event)


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

    def __init__(self, mode, input_dir=None, output_dir=None, white_trigger=235,
                 selected_points_map=None, color_tolerance=5, do_remove_white=True,
                 do_remove_points=True, do_resize=False, target_width=0, target_height=0,
                 keep_original_name=False, white_roi_map=None, roi_remove_white=False,
                 min_white_block_side=2, video_paths=None, frame_interval=1,
                 frame_mode="interval", target_frame_count=None,
                 rename_prefix='Output', rename_start_num=1,
                 min_component_area=100, merge_strokes_map=None,
                 split_strokes_map=None, parent=None):
        super().__init__(parent)
        self.mode = mode  # "batch_process"、"video_extract" 或 "batch_rename"
        # 批量重命名参数
        self.rename_prefix = rename_prefix
        self.rename_start_num = rename_start_num
        self.min_component_area = min_component_area
        self.merge_strokes_map = merge_strokes_map or {}
        self.split_strokes_map = split_strokes_map or {}
        # 批量抠图参数
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.white_trigger = white_trigger
        self.selected_points_map = selected_points_map or {}
        self.color_tolerance = color_tolerance
        self.do_remove_white = do_remove_white
        self.do_remove_points = do_remove_points
        self.do_resize = do_resize
        self.target_width = target_width
        self.target_height = target_height
        self.keep_original_name = keep_original_name
        self.white_roi_map = white_roi_map or {}
        self.roi_remove_white = roi_remove_white
        self.min_white_block_side = min_white_block_side
        # 视频抽帧参数（支持多个视频）
        self.video_paths = video_paths or []  # 视频文件路径列表
        self.frame_interval = frame_interval
        self.frame_mode = frame_mode  # "interval" 或 "count"
        self.target_frame_count = target_frame_count

    def run(self):
        def _on_progress(current, total, input_path, output_path, ok):
            self.progress.emit(current, total, input_path, output_path, ok)

        try:
            if self.mode == "batch_process":
                self.log.emit("开始批量抠图...")
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
                    white_roi_map=self.white_roi_map,
                    roi_remove_white=self.roi_remove_white,
                    min_white_block_side=self.min_white_block_side,
                    on_progress=_on_progress,
                )
                self.log.emit("批量抠图结束")
                self.finished.emit(success_count, total_count)
            elif self.mode == "object_slice":
                self.log.emit("开始一键切图...")
                success_count, total_count, object_count = slice_directory_objects(
                    self.input_dir,
                    self.output_dir,
                    min_component_area=self.min_component_area,
                    white_trigger=self.white_trigger,
                    selected_points_map=self.selected_points_map,
                    merge_strokes_map=self.merge_strokes_map,
                    split_strokes_map=self.split_strokes_map,
                    color_tolerance=self.color_tolerance,
                    do_remove_white=self.do_remove_white,
                    do_remove_points=self.do_remove_points,
                    do_resize=self.do_resize,
                    target_width=self.target_width,
                    target_height=self.target_height,
                    on_progress=_on_progress,
                )
                self.log.emit(f"一键切图结束，共导出 {object_count} 个物件")
                self.finished.emit(success_count, total_count)
            elif self.mode == "video_extract":
                self.log.emit("开始视频抽帧...")
                total_saved = 0
                total_expected = 0
                for video_path in self.video_paths:
                    if not os.path.isfile(video_path):
                        self.log.emit(f"跳过无效视频文件: {video_path}")
                        continue
                    self.log.emit(f"处理视频: {os.path.basename(video_path)}")
                    if self.frame_mode == "count" and self.target_frame_count:
                        # 按目标帧数抽帧
                        saved_count, expected, frame_subdir = extract_video_frames(
                            video_path,
                            self.output_dir,
                            target_frame_count=self.target_frame_count,
                            output_format="png",
                            on_progress=_on_progress,
                        )
                    else:
                        # 按间隔抽帧
                        saved_count, expected, frame_subdir = extract_video_frames(
                            video_path,
                            self.output_dir,
                            interval_seconds=self.frame_interval,
                            output_format="png",
                            on_progress=_on_progress,
                        )
                    total_saved += saved_count
                    total_expected += expected
                    if frame_subdir:
                        self.log.emit(f"  帧已保存到: {frame_subdir}/")
                self.log.emit(f"视频抽帧结束，共处理 {len(self.video_paths)} 个视频，保存 {total_saved} 帧")
                self.finished.emit(total_saved, total_expected)
            elif self.mode == 'batch_rename':
                self.log.emit("开始批量重命名...")
                success, total = batch_rename_images(
                    self.input_dir,
                    self.rename_prefix,
                    self.rename_start_num,
                    on_progress=lambda cur, tot, old, new, ok:
                        self.progress.emit(cur, tot, old, new, ok)
                )
                self.log.emit(f"批量重命名结束，共重命名 {success}/{total} 个文件")
                self.finished.emit(success, total)
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
        self.merge_strokes_map = {}
        self.split_strokes_map = {}
        self.white_roi_map = {}
        self.current_drag_stroke = None
        self.is_drawing_stroke = False
        self.stroke_mode_enabled = False
        self.roi_mode_enabled = False
        self.roi_erase_enabled = False
        self.roi_drag_start = None
        self.current_drag_roi = None
        self.current_input_path = None
        self.current_input_key = None
        self.current_image_size = None
        self.preview_original_pixmap = None
        self.preview_display_rect = None

        central = QWidget(self)
        layout = QGridLayout(central)
        self.main_layout = layout

        # 功能选择
        self.mode_combo = QComboBox()
        self.mode_combo.addItem("批量抠图", "batch_process")
        self.mode_combo.addItem("一键切图", "object_slice")
        self.mode_combo.addItem("视频抽帧", "video_extract")
        self.mode_combo.addItem("批量重命名", "batch_rename")
        self.mode_combo.currentIndexChanged.connect(self.on_mode_changed)

        # 抽帧间隔设置
        self.frame_interval_spin = QSpinBox()
        self.frame_interval_spin.setRange(1, 3600)
        self.frame_interval_spin.setValue(1)
        self.frame_interval_spin.setSuffix(" 秒")
        self.frame_interval_label = QLabel("抽帧间隔")

        # 目标帧数设置
        self.target_frame_count_spin = QSpinBox()
        self.target_frame_count_spin.setRange(1, 99999)
        self.target_frame_count_spin.setValue(30)
        self.target_frame_count_label = QLabel("目标帧数")

        # 抽帧方式选择（默认按数量抽帧）
        self.frame_mode_combo = QComboBox()
        self.frame_mode_combo.addItem("按数量抽帧", "count")
        self.frame_mode_combo.addItem("按间隔抽帧", "interval")
        self.frame_mode_combo.currentIndexChanged.connect(self.on_frame_mode_changed)

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
        self.toggle_stroke_mode_button = QPushButton("开启划线模式")
        self.toggle_stroke_mode_button.setCheckable(True)

        self.remove_white_checkbox = QCheckBox("去除白边")
        self.remove_white_checkbox.setChecked(True)
        self.remove_points_checkbox = QCheckBox("选点删除")
        self.remove_points_checkbox.setChecked(True)
        self.resize_checkbox = QCheckBox("调整分辨率")
        self.resize_checkbox.setChecked(False)
        self.keep_name_checkbox = QCheckBox("保持原文件名")
        self.keep_name_checkbox.setChecked(False)
        self.roi_remove_white_checkbox = QCheckBox("区域去白")
        self.roi_remove_white_checkbox.setChecked(False)
        self.white_block_side_label = QLabel("最小白块边长")
        self.white_block_side_spin = QSpinBox()
        self.white_block_side_spin.setRange(1, 128)
        self.white_block_side_spin.setValue(2)
        self.toggle_roi_mode_button = QPushButton("开启框选模式")
        self.toggle_roi_mode_button.setCheckable(True)
        self.toggle_roi_erase_button = QPushButton("关闭选区擦除")
        self.toggle_roi_erase_button.setCheckable(True)

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

        base_dir = self._base_dir()
        default_input = os.path.join(base_dir, "Input")
        default_output = os.path.join(base_dir, "OutPut")
        self._video_input_dir = os.path.join(base_dir, "VideoInput")
        self._rename_target_dir = os.path.join(base_dir, "OutPut")
        self.input_edit.setText(default_input)
        self.output_edit.setText(default_output)

        # 功能选择行
        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel("功能选择"))
        mode_row.addWidget(self.mode_combo)
        mode_row.addWidget(self.frame_mode_combo)
        mode_row.addWidget(self.frame_interval_label)
        mode_row.addWidget(self.frame_interval_spin)
        mode_row.addWidget(self.target_frame_count_label)
        mode_row.addWidget(self.target_frame_count_spin)
        mode_row.addStretch(1)
        layout.addLayout(mode_row, 0, 0, 1, 4)

        # 视频抽帧控件默认隐藏
        self.frame_mode_combo.setVisible(False)
        self.frame_interval_label.setVisible(False)
        self.frame_interval_spin.setVisible(False)
        self.target_frame_count_label.setVisible(False)
        self.target_frame_count_spin.setVisible(False)

        layout.addWidget(self.preview_label, 1, 0, 1, 4)
        layout.addWidget(self.progress_bar, 2, 0, 1, 4)

        layout.setRowStretch(1, 1)
        layout.setRowStretch(6, 0)
        layout.setColumnStretch(1, 1)
        layout.setColumnStretch(0, 0)
        layout.setColumnStretch(2, 0)
        layout.setColumnStretch(3, 0)

        self.input_label = QLabel("输入目录")
        layout.addWidget(self.input_label, 3, 0)
        layout.addWidget(self.input_edit, 3, 1)
        layout.addWidget(self.input_button, 3, 2)

        point_row = QHBoxLayout()
        point_row.addWidget(self.image_button)
        point_row.addWidget(self.toggle_stroke_mode_button)
        point_row.addWidget(self.toggle_roi_mode_button)
        point_row.addWidget(self.toggle_roi_erase_button)
        point_row.addWidget(self.clear_points_button)
        layout.addLayout(point_row, 3, 3)

        self.output_label = QLabel("输出目录")
        layout.addWidget(self.output_label, 4, 0)
        layout.addWidget(self.output_edit, 4, 1, 1, 2)
        layout.addWidget(self.output_button, 4, 3)

        self.threshold_label = QLabel("白色阈值")
        layout.addWidget(self.threshold_label, 5, 0)
        layout.addWidget(self.threshold_spin, 5, 1)

        self.tolerance_label = QLabel("选点容差")
        layout.addWidget(self.tolerance_label, 5, 2)
        layout.addWidget(self.tolerance_spin, 5, 3)

        self.component_area_label = QLabel("最小连通像素")
        self.component_area_spin = QSpinBox()
        self.component_area_spin.setRange(100, 99999999)
        self.component_area_spin.setValue(100)
        self.component_area_label.hide()
        self.component_area_spin.hide()

        self.stroke_mode_label = QLabel("划线操作")
        self.stroke_mode_combo = QComboBox()
        self.stroke_mode_combo.addItem("连接物件", "merge")
        self.stroke_mode_combo.addItem("切开物件", "split")
        self.stroke_mode_combo.currentIndexChanged.connect(self.on_stroke_mode_changed)
        self.stroke_mode_label.hide()
        self.stroke_mode_combo.hide()
        self.roi_remove_white_checkbox.hide()
        self.toggle_roi_mode_button.hide()
        self.white_block_side_label.hide()
        self.white_block_side_spin.hide()
        self.toggle_roi_erase_button.hide()

        # 批量重命名专属控件（与 row 5 复用行，默认隐藏）
        self.rename_prefix_label = QLabel("文件前缀")
        self.rename_prefix_edit = QLineEdit("Output")
        self.rename_start_label = QLabel("起始序号")
        self.rename_start_spin = QSpinBox()
        self.rename_start_spin.setRange(0, 99999)
        self.rename_start_spin.setValue(1)
        layout.addWidget(self.rename_prefix_label, 5, 0)
        layout.addWidget(self.rename_prefix_edit, 5, 1)
        layout.addWidget(self.rename_start_label, 5, 2)
        layout.addWidget(self.rename_start_spin, 5, 3)
        self.rename_prefix_label.hide()
        self.rename_prefix_edit.hide()
        self.rename_start_label.hide()
        self.rename_start_spin.hide()

        option_row = QHBoxLayout()
        option_row.addWidget(self.remove_white_checkbox)
        option_row.addWidget(self.remove_points_checkbox)
        option_row.addWidget(self.roi_remove_white_checkbox)
        option_row.addWidget(self.resize_checkbox)
        option_row.addWidget(self.keep_name_checkbox)
        option_row.addWidget(self.white_block_side_label)
        option_row.addWidget(self.white_block_side_spin)
        option_row.addWidget(self.component_area_label)
        option_row.addWidget(self.component_area_spin)
        option_row.addWidget(self.stroke_mode_label)
        option_row.addWidget(self.stroke_mode_combo)
        layout.addLayout(option_row, 6, 0, 1, 4)

        resize_row = QHBoxLayout()
        self.resize_width_label = QLabel("宽")
        self.resize_height_label = QLabel("高")
        resize_row.addWidget(self.resize_width_label)
        resize_row.addWidget(self.resize_width_spin)
        resize_row.addWidget(self.resize_height_label)
        resize_row.addWidget(self.resize_height_spin)
        resize_row.addStretch(1)
        layout.addLayout(resize_row, 7, 0, 1, 4)

        button_row = QHBoxLayout()
        button_row.addWidget(self.start_button)
        layout.addLayout(button_row, 8, 0, 1, 4)

        self.setCentralWidget(central)

        self.input_button.clicked.connect(self.select_input_dir)
        self.output_button.clicked.connect(self.select_output_dir)
        self.image_button.clicked.connect(self.select_preview_image)
        self.toggle_stroke_mode_button.clicked.connect(self.toggle_stroke_mode)
        self.toggle_roi_mode_button.clicked.connect(self.toggle_roi_mode)
        self.toggle_roi_erase_button.clicked.connect(self.toggle_roi_erase_mode)
        self.start_button.clicked.connect(self.start_processing)
        self.clear_points_button.clicked.connect(self.clear_selected_points)
        self.preview_label.pressed.connect(self.on_preview_pressed)
        self.preview_label.moved.connect(self.on_preview_moved)
        self.preview_label.released.connect(self.on_preview_released)

        # 程序启动时按当前模式刷新控件可见性
        self.on_mode_changed(self.mode_combo.currentIndex())

        # 程序启动时自动加载默认输入目录的第一张图片
        default_input_dir = self.input_edit.text().strip()
        if default_input_dir and os.path.isdir(default_input_dir):
            first_image = self._get_first_image_in_directory(default_input_dir)
            if first_image:
                self._show_preview(first_image)

    def toggle_stroke_mode(self, checked):
        self.stroke_mode_enabled = checked
        if not checked:
            self.current_drag_stroke = None
            self.is_drawing_stroke = False
        self._sync_stroke_mode_ui()

    def on_stroke_mode_changed(self, index):
        self._sync_stroke_mode_ui()

    def _sync_stroke_mode_ui(self):
        enabled = self.mode_combo.currentData() == "object_slice" and self.stroke_mode_enabled
        self.toggle_stroke_mode_button.setChecked(enabled)
        self.toggle_stroke_mode_button.setText("关闭划线模式" if enabled else "开启划线模式")
        self.preview_label.setCursor(Qt.CrossCursor if enabled else Qt.ArrowCursor)
        self._refresh_preview_scaled()

    def toggle_roi_mode(self, checked):
        self.roi_mode_enabled = checked
        if not checked:
            self.roi_drag_start = None
            self.current_drag_roi = None
        self._sync_roi_mode_ui()

    def _sync_roi_mode_ui(self):
        enabled = self.mode_combo.currentData() == "batch_process" and self.roi_mode_enabled
        self.toggle_roi_mode_button.setChecked(enabled)
        self.toggle_roi_mode_button.setText("关闭框选模式" if enabled else "开启框选模式")
        self._sync_roi_erase_mode_ui()
        self._refresh_preview_scaled()

    def toggle_roi_erase_mode(self, checked):
        self.roi_erase_enabled = checked
        self._sync_roi_erase_mode_ui()

    def _sync_roi_erase_mode_ui(self):
        visible = self.mode_combo.currentData() == "batch_process" and self.roi_mode_enabled
        self.toggle_roi_erase_button.setVisible(visible)
        enabled = visible and self.roi_erase_enabled
        self.toggle_roi_erase_button.setChecked(enabled)
        self.toggle_roi_erase_button.setText("开启选区擦除" if enabled else "关闭选区擦除")
        if enabled:
            self.preview_label.setCursor(Qt.ForbiddenCursor)
        elif visible:
            self.preview_label.setCursor(Qt.CrossCursor)
        else:
            self.preview_label.setCursor(Qt.ArrowCursor)

    def on_frame_mode_changed(self, index):
        """抽帧方式切换时的回调"""
        mode = self.frame_mode_combo.currentData()
        if mode == "interval":
            # 按间隔抽帧
            self.frame_interval_label.setVisible(True)
            self.frame_interval_spin.setVisible(True)
            self.target_frame_count_label.setVisible(False)
            self.target_frame_count_spin.setVisible(False)
        else:
            # 按数量抽帧
            self.frame_interval_label.setVisible(False)
            self.frame_interval_spin.setVisible(False)
            self.target_frame_count_label.setVisible(True)
            self.target_frame_count_spin.setVisible(True)

    def _set_output_row_visible(self, visible):
        fn = lambda w: w.show() if visible else w.hide()
        for w in [self.output_label, self.output_edit, self.output_button]:
            fn(w)

    def on_mode_changed(self, index):
        """功能模式切换时的回调"""
        mode = self.mode_combo.currentData()
        if mode == "batch_process":
            # 批量抠图模式
            self.input_label.setText("输入目录")
            self.input_button.setText("选择输入目录")
            self.input_button.clicked.disconnect()
            self.input_button.clicked.connect(self.select_input_dir)
            self.input_edit.setText(self._last_batch_input_dir if hasattr(self, '_last_batch_input_dir') else os.path.join(self._base_dir(), "Input"))
            self.output_edit.setText(os.path.join(self._base_dir(), "OutPut"))
            self.clear_points_button.setText("清除选点")
            self.toggle_stroke_mode_button.setChecked(False)
            self.stroke_mode_enabled = False
            self.toggle_roi_mode_button.setChecked(False)
            self.roi_mode_enabled = False
            self.roi_erase_enabled = False
            self.toggle_roi_erase_button.setChecked(False)
            self.roi_drag_start = None
            self.current_drag_roi = None
            self._sync_roi_mode_ui()
            self._sync_stroke_mode_ui()

            # 显示输出目录行
            self._set_output_row_visible(True)
            # 显示批量抠图相关控件
            self.threshold_label.setVisible(True)
            self.threshold_spin.setVisible(True)
            self.tolerance_label.setVisible(True)
            self.tolerance_spin.setVisible(True)
            self.component_area_label.setVisible(False)
            self.component_area_spin.setVisible(False)
            self.stroke_mode_label.setVisible(False)
            self.stroke_mode_combo.setVisible(False)
            self.remove_white_checkbox.setVisible(True)
            self.remove_points_checkbox.setVisible(True)
            self.roi_remove_white_checkbox.setVisible(True)
            self.white_block_side_label.setVisible(True)
            self.white_block_side_spin.setVisible(True)
            self.resize_checkbox.setVisible(True)
            self.keep_name_checkbox.setVisible(True)
            # 根据复选框状态显示/隐藏分辨率设置
            self.on_resize_toggled(self.resize_checkbox.isChecked())
            self.resize_width_label.setVisible(True)
            self.resize_width_spin.setVisible(True)
            self.resize_height_label.setVisible(True)
            self.resize_height_spin.setVisible(True)
            self.image_button.setVisible(True)
            self.toggle_stroke_mode_button.setVisible(False)
            self.toggle_roi_mode_button.setVisible(True)
            self.toggle_roi_erase_button.setVisible(self.roi_mode_enabled)
            self.clear_points_button.setVisible(True)

            # 隐藏视频抽帧相关控件
            self.frame_mode_combo.setVisible(False)
            self.frame_interval_label.setVisible(False)
            self.frame_interval_spin.setVisible(False)
            self.target_frame_count_label.setVisible(False)
            self.target_frame_count_spin.setVisible(False)
            # 隐藏重命名专属控件
            for w in [self.rename_prefix_label, self.rename_prefix_edit,
                      self.rename_start_label, self.rename_start_spin]:
                w.setVisible(False)
        elif mode == "object_slice":
            if self.input_label.text() == "输入目录":
                self._last_batch_input_dir = self.input_edit.text()

            self.input_label.setText("输入目录")
            self.input_button.setText("选择输入目录")
            self.input_button.clicked.disconnect()
            self.input_button.clicked.connect(self.select_input_dir)
            self.input_edit.setText(self._last_batch_input_dir if hasattr(self, '_last_batch_input_dir') else os.path.join(self._base_dir(), "Input"))
            self.output_edit.setText(os.path.join(self._base_dir(), "OutPut"))
            self.clear_points_button.setText("清除当前线")
            self.toggle_stroke_mode_button.setChecked(False)
            self.stroke_mode_enabled = False
            self.toggle_roi_mode_button.setChecked(False)
            self.roi_mode_enabled = False
            self.roi_erase_enabled = False
            self.toggle_roi_erase_button.setChecked(False)
            self.roi_drag_start = None
            self.current_drag_roi = None
            self._sync_roi_mode_ui()
            self._sync_stroke_mode_ui()

            self._set_output_row_visible(True)
            self.threshold_label.setVisible(False)
            self.threshold_spin.setVisible(False)
            self.tolerance_label.setVisible(False)
            self.tolerance_spin.setVisible(False)
            self.component_area_label.setVisible(True)
            self.component_area_spin.setVisible(True)
            self.stroke_mode_label.setVisible(True)
            self.stroke_mode_combo.setVisible(True)
            self.remove_white_checkbox.setVisible(False)
            self.remove_points_checkbox.setVisible(False)
            self.roi_remove_white_checkbox.setVisible(False)
            self.white_block_side_label.setVisible(False)
            self.white_block_side_spin.setVisible(False)
            self.resize_checkbox.setVisible(False)
            self.keep_name_checkbox.setVisible(False)
            self.resize_width_label.setVisible(False)
            self.resize_width_spin.setVisible(False)
            self.resize_height_label.setVisible(False)
            self.resize_height_spin.setVisible(False)
            self.image_button.setVisible(True)
            self.toggle_stroke_mode_button.setVisible(True)
            self.toggle_roi_mode_button.setVisible(False)
            self.toggle_roi_erase_button.setVisible(False)
            self.clear_points_button.setVisible(True)

            self.frame_mode_combo.setVisible(False)
            self.frame_interval_label.setVisible(False)
            self.frame_interval_spin.setVisible(False)
            self.target_frame_count_label.setVisible(False)
            self.target_frame_count_spin.setVisible(False)
            for w in [self.rename_prefix_label, self.rename_prefix_edit,
                      self.rename_start_label, self.rename_start_spin]:
                w.setVisible(False)
        elif mode == "video_extract":
            # 保存当前批量抠图的输入目录
            if self.input_label.text() == "输入目录":
                self._last_batch_input_dir = self.input_edit.text()

            # 视频抽帧模式
            self.input_label.setText("视频输入")
            self.input_button.setText("选择视频/目录")
            self.input_button.clicked.disconnect()
            self.input_button.clicked.connect(self.select_input_video_or_dir)
            # 视频模式下，默认输入目录为 VideoInput
            if hasattr(self, '_video_input_dir'):
                self.input_edit.setText(self._video_input_dir)
            self.output_edit.setText(os.path.join(self._base_dir(), "Input"))

            # 显示输出目录行
            self._set_output_row_visible(True)
            # 隐藏批量抠图相关控件
            self.threshold_label.setVisible(False)
            self.threshold_spin.setVisible(False)
            self.tolerance_label.setVisible(False)
            self.tolerance_spin.setVisible(False)
            self.component_area_label.setVisible(False)
            self.component_area_spin.setVisible(False)
            self.stroke_mode_label.setVisible(False)
            self.stroke_mode_combo.setVisible(False)
            self.remove_white_checkbox.setVisible(False)
            self.remove_points_checkbox.setVisible(False)
            self.roi_remove_white_checkbox.setVisible(False)
            self.white_block_side_label.setVisible(False)
            self.white_block_side_spin.setVisible(False)
            self.resize_checkbox.setVisible(False)
            self.keep_name_checkbox.setVisible(False)
            self.resize_width_label.setVisible(False)
            self.resize_width_spin.setVisible(False)
            self.resize_height_label.setVisible(False)
            self.resize_height_spin.setVisible(False)
            self.image_button.setVisible(False)
            self.toggle_stroke_mode_button.setVisible(False)
            self.toggle_roi_mode_button.setVisible(False)
            self.toggle_roi_erase_button.setVisible(False)
            self.clear_points_button.setVisible(False)
            # 隐藏重命名专属控件
            for w in [self.rename_prefix_label, self.rename_prefix_edit,
                      self.rename_start_label, self.rename_start_spin]:
                w.setVisible(False)

            # 显示视频抽帧相关控件
            self.frame_mode_combo.setVisible(True)
            # 根据当前抽帧方式显示对应控件（默认按数量抽帧）
            self.on_frame_mode_changed(self.frame_mode_combo.currentIndex())
        elif mode == "batch_rename":
            # 保存当前批量抠图的输入目录
            if self.input_label.text() == "输入目录":
                self._last_batch_input_dir = self.input_edit.text()

            # 批量重命名模式
            self.input_label.setText("目标目录")
            self.input_button.setText("选择目录")
            self.input_button.clicked.disconnect()
            self.input_button.clicked.connect(self.select_input_dir)
            self.input_edit.setText(self._rename_target_dir)

            # 隐藏输出目录行（重命名为原地操作）
            self._set_output_row_visible(False)
            # 隐藏批量抠图相关控件
            for w in [self.threshold_label, self.threshold_spin,
                      self.tolerance_label, self.tolerance_spin,
                      self.component_area_label, self.component_area_spin,
                      self.stroke_mode_label, self.stroke_mode_combo,
                      self.remove_white_checkbox, self.remove_points_checkbox,
                      self.roi_remove_white_checkbox, self.white_block_side_label, self.white_block_side_spin,
                      self.resize_checkbox, self.keep_name_checkbox,
                      self.resize_width_label, self.resize_width_spin,
                      self.resize_height_label, self.resize_height_spin,
                      self.image_button, self.toggle_stroke_mode_button, self.toggle_roi_mode_button, self.toggle_roi_erase_button, self.clear_points_button]:
                w.setVisible(False)
            # 隐藏视频抽帧相关控件
            for w in [self.frame_mode_combo, self.frame_interval_label,
                      self.frame_interval_spin, self.target_frame_count_label,
                      self.target_frame_count_spin]:
                w.setVisible(False)
            # 显示重命名专属控件
            for w in [self.rename_prefix_label, self.rename_prefix_edit,
                      self.rename_start_label, self.rename_start_spin]:
                w.setVisible(True)

    def _base_dir(self):
        return os.path.dirname(
            os.path.abspath(sys.executable if getattr(sys, "frozen", False) else __file__)
        )

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
        mode = self.mode_combo.currentData()
        if mode == "object_slice":
            current_mode = self.stroke_mode_combo.currentData()
            if current_mode == "split":
                strokes = self.split_strokes_map.get(self.current_input_key)
            else:
                strokes = self.merge_strokes_map.get(self.current_input_key)
            if strokes:
                strokes.clear()
            self.current_drag_stroke = None
            self.is_drawing_stroke = False
            self._refresh_preview_scaled()
            return
        if mode == "batch_process" and self.roi_mode_enabled:
            rois = self.white_roi_map.get(self.current_input_key)
            if rois:
                rois.clear()
            self.roi_drag_start = None
            self.current_drag_roi = None
            self._refresh_preview_scaled()
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

    def _current_merge_strokes(self):
        if not self.current_input_key:
            return []
        return self.merge_strokes_map.setdefault(self.current_input_key, [])

    def _current_split_strokes(self):
        if not self.current_input_key:
            return []
        return self.split_strokes_map.setdefault(self.current_input_key, [])

    def _normalize_roi(self, start_point, end_point):
        x1, y1 = start_point
        x2, y2 = end_point
        left = min(x1, x2)
        top = min(y1, y2)
        width = abs(x2 - x1) + 1
        height = abs(y2 - y1) + 1
        return left, top, width, height

    def _rects_intersect(self, rect_a, rect_b):
        ax, ay, aw, ah = rect_a
        bx, by, bw, bh = rect_b
        ar = ax + aw
        ab = ay + ah
        br = bx + bw
        bb = by + bh
        return not (ar <= bx or br <= ax or ab <= by or bb <= ay)

    def _current_white_rois(self):
        if not self.current_input_key:
            return []
        return self.white_roi_map.setdefault(self.current_input_key, [])

    def _preview_position_to_image_point(self, position):
        if not self.preview_display_rect or not self.current_image_size:
            return None
        if not self.preview_display_rect.contains(position):
            return None

        img_w, img_h = self.current_image_size
        rect = self.preview_display_rect
        if rect.width() <= 0 or rect.height() <= 0:
            return None

        x = (position.x() - rect.x()) * img_w / rect.width()
        y = (position.y() - rect.y()) * img_h / rect.height()
        img_x = int(round(x))
        img_y = int(round(y))

        if img_x < 0 or img_y < 0 or img_x >= img_w or img_y >= img_h:
            return None
        return img_x, img_y

    def _append_point_to_current_stroke(self, point):
        if not self.current_drag_stroke:
            self.current_drag_stroke = [point]
            return
        last_x, last_y = self.current_drag_stroke[-1]
        if (last_x - point[0]) ** 2 + (last_y - point[1]) ** 2 >= 4:
            self.current_drag_stroke.append(point)

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
        mode = self.mode_combo.currentData()
        if mode == "object_slice":
            return
        if mode == "batch_process" and self.roi_mode_enabled:
            if button == Qt.RightButton and self.current_input_key:
                rois = self.white_roi_map.get(self.current_input_key)
                if rois:
                    rois.pop()
                self.current_drag_roi = None
                self.roi_drag_start = None
                self._refresh_preview_scaled()
            return
        point = self._preview_position_to_image_point(position)
        if point is None:
            return

        img_x, img_y = point
        points = self._current_selected_points()
        if button == Qt.RightButton:
            self._remove_nearest_point(points, img_x, img_y, radius=8)
        else:
            points.append((img_x, img_y))
        self._refresh_preview_scaled()

    def on_preview_pressed(self, position, button):
        mode = self.mode_combo.currentData()
        if mode == "batch_process" and self.roi_mode_enabled:
            if button != Qt.LeftButton:
                self.on_preview_clicked(position, button)
                return
            point = self._preview_position_to_image_point(position)
            if point is None:
                return
            self.roi_drag_start = point
            self.current_drag_roi = (point[0], point[1], 1, 1)
            self._refresh_preview_scaled()
            return
        if mode != "object_slice" or button != Qt.LeftButton:
            self.on_preview_clicked(position, button)
            return
        if not self.stroke_mode_enabled:
            return
        point = self._preview_position_to_image_point(position)
        if point is None:
            return
        self.is_drawing_stroke = True
        self.current_drag_stroke = [point]
        self._refresh_preview_scaled()

    def on_preview_moved(self, position, button):
        mode = self.mode_combo.currentData()
        if mode == "batch_process" and self.roi_mode_enabled:
            if not self.roi_drag_start or button != Qt.LeftButton:
                return
            point = self._preview_position_to_image_point(position)
            if point is None:
                return
            self.current_drag_roi = self._normalize_roi(self.roi_drag_start, point)
            self._refresh_preview_scaled()
            return
        if mode != "object_slice":
            return
        if not self.stroke_mode_enabled:
            return
        if not self.is_drawing_stroke or button != Qt.LeftButton:
            return
        point = self._preview_position_to_image_point(position)
        if point is None:
            return
        self._append_point_to_current_stroke(point)
        self._refresh_preview_scaled()

    def on_preview_released(self, position, button):
        mode = self.mode_combo.currentData()
        if mode == "batch_process" and self.roi_mode_enabled:
            if button != Qt.LeftButton or not self.roi_drag_start:
                return
            point = self._preview_position_to_image_point(position)
            if point is not None:
                roi = self._normalize_roi(self.roi_drag_start, point)
            else:
                roi = self.current_drag_roi
            self.roi_drag_start = None
            self.current_drag_roi = None
            if roi and self.current_input_key:
                rois = self._current_white_rois()
                if self.roi_erase_enabled:
                    self.white_roi_map[self.current_input_key] = [
                        existing for existing in rois if not self._rects_intersect(existing, roi)
                    ]
                else:
                    rois.append(roi)
            self._refresh_preview_scaled()
            return
        if mode != "object_slice" or button != Qt.LeftButton:
            return
        if not self.stroke_mode_enabled:
            return
        if not self.is_drawing_stroke:
            return
        point = self._preview_position_to_image_point(position)
        if point is not None:
            self._append_point_to_current_stroke(point)
        stroke = self.current_drag_stroke or []
        if len(stroke) >= 2:
            if self.stroke_mode_combo.currentData() == "split":
                self._current_split_strokes().append(stroke.copy())
            else:
                self._current_merge_strokes().append(stroke.copy())
        self.current_drag_stroke = None
        self.is_drawing_stroke = False
        self._refresh_preview_scaled()

    def _get_video_files_in_directory(self, dir_path):
        """扫描目录下的视频文件，返回所有视频文件路径列表"""
        if not dir_path or not os.path.isdir(dir_path):
            return []
        video_extensions = ('.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm')
        video_files = []
        try:
            for name in os.listdir(dir_path):
                if name.lower().endswith(video_extensions):
                    full_path = os.path.join(dir_path, name)
                    if os.path.isfile(full_path):
                        video_files.append(full_path)
        except Exception:
            pass
        return sorted(video_files)

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

    def select_input_video(self):
        # 视频模式默认从 VideoInput 目录开始
        if hasattr(self, '_video_input_dir') and os.path.isdir(self._video_input_dir):
            start_dir = self._video_input_dir
        else:
            start_dir = os.path.dirname(self.input_edit.text()) if self.input_edit.text() else self._base_dir()
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择视频文件",
            start_dir,
            "Video Files (*.mp4 *.avi *.mov *.mkv *.flv *.wmv *.webm);;All Files (*)",
        )
        if file_path:
            self.input_edit.setText(file_path)

    def select_input_video_or_dir(self):
        """选择视频文件或目录（供视频抽帧模式使用）"""
        # 先尝试选择目录
        current_path = self.input_edit.text().strip()
        if current_path and os.path.isdir(current_path):
            start_dir = current_path
        elif hasattr(self, '_video_input_dir') and os.path.isdir(self._video_input_dir):
            start_dir = self._video_input_dir
        else:
            start_dir = self._base_dir()

        selected_dir = QFileDialog.getExistingDirectory(self, "选择视频目录（或取消选择单个视频）", start_dir)
        if selected_dir:
            self.input_edit.setText(selected_dir)
        else:
            # 用户取消目录选择，则选择单个视频文件
            file_path, _ = QFileDialog.getOpenFileName(
                self,
                "选择视频文件",
                start_dir,
                "Video Files (*.mp4 *.avi *.mov *.mkv *.flv *.wmv *.webm);;All Files (*)",
            )
            if file_path:
                self.input_edit.setText(file_path)

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
        mode = self.mode_combo.currentData()
        input_path = self.input_edit.text().strip()
        output_dir = self.output_edit.text().strip()

        if mode != "batch_rename" and (not output_dir or not os.path.isdir(output_dir)):
            if self.log_dialog is None:
                self.log_dialog = LogDialog(self)
            self.log_dialog.append("请输入有效的输出目录")
            self.log_dialog.show()
            return

        self.current_drag_stroke = None
        self.is_drawing_stroke = False
        self.roi_drag_start = None
        self.current_drag_roi = None
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
        if self.image_button.isVisible():
            self.image_button.setEnabled(False)
        if self.clear_points_button.isVisible():
            self.clear_points_button.setEnabled(False)

        if self.log_dialog is None:
            self.log_dialog = LogDialog(self)
        self.log_dialog.log_edit.clear()
        self.log_dialog.append("开始处理...")
        self.log_dialog.show()
        self.log_dialog.raise_()

        if mode == "batch_process":
            input_dir = input_path
            if not input_dir or not os.path.isdir(input_dir):
                self.log_dialog.append("请输入有效的输入目录")
                self.start_button.setEnabled(True)
                self.input_button.setEnabled(True)
                self.output_button.setEnabled(True)
                if self.image_button.isVisible():
                    self.image_button.setEnabled(True)
                if self.clear_points_button.isVisible():
                    self.clear_points_button.setEnabled(True)
                return

            if self.current_input_path and input_dir and not os.path.commonpath([self.current_input_path, input_dir]) == input_dir:
                self.log_dialog.append("当前预览图片不在输入目录内，将按文件名匹配输入目录图片")

            white_trigger = self.threshold_spin.value()
            color_tolerance = self.tolerance_spin.value()
            selected_points_snapshot = self._resolve_selected_points(input_dir)
            white_roi_snapshot = self._resolve_white_rois(input_dir)
            if not selected_points_snapshot:
                self.log_dialog.append("提示: 当前没有选点，未传入选点清理")

            self.worker = ProcessWorker(
                mode="batch_process",
                input_dir=input_dir,
                output_dir=output_dir,
                white_trigger=white_trigger,
                selected_points_map=selected_points_snapshot,
                white_roi_map=white_roi_snapshot,
                roi_remove_white=self.roi_remove_white_checkbox.isChecked(),
                min_white_block_side=self.white_block_side_spin.value(),
                color_tolerance=color_tolerance,
                do_remove_white=self.remove_white_checkbox.isChecked(),
                do_remove_points=self.remove_points_checkbox.isChecked(),
                do_resize=self.resize_checkbox.isChecked(),
                target_width=self.resize_width_spin.value(),
                target_height=self.resize_height_spin.value(),
                keep_original_name=self.keep_name_checkbox.isChecked(),
                parent=self,
            )
        elif mode == "object_slice":
            input_dir = input_path
            if not input_dir or not os.path.isdir(input_dir):
                self.log_dialog.append("请输入有效的输入目录")
                self.start_button.setEnabled(True)
                self.input_button.setEnabled(True)
                self.output_button.setEnabled(True)
                if self.image_button.isVisible():
                    self.image_button.setEnabled(True)
                if self.clear_points_button.isVisible():
                    self.clear_points_button.setEnabled(True)
                return

            merge_strokes_snapshot = self._resolve_merge_strokes(input_dir)
            split_strokes_snapshot = self._resolve_split_strokes(input_dir)
            self.worker = ProcessWorker(
                mode="object_slice",
                input_dir=input_dir,
                output_dir=output_dir,
                white_trigger=235,
                selected_points_map=None,
                color_tolerance=5,
                do_remove_white=True,
                do_remove_points=False,
                do_resize=False,
                target_width=0,
                target_height=0,
                min_component_area=self.component_area_spin.value(),
                merge_strokes_map=merge_strokes_snapshot,
                split_strokes_map=split_strokes_snapshot,
                parent=self,
            )
        elif mode == "video_extract":
            video_files = []
            if not input_path:
                # 使用默认 VideoInput 目录
                if hasattr(self, '_video_input_dir') and os.path.isdir(self._video_input_dir):
                    input_path = self._video_input_dir
                else:
                    self.log_dialog.append("请选择视频文件或输入目录")
                    self.start_button.setEnabled(True)
                    self.input_button.setEnabled(True)
                    self.output_button.setEnabled(True)
                    return

            if os.path.isfile(input_path):
                # 单个视频文件
                video_files = [input_path]
            elif os.path.isdir(input_path):
                # 目录，扫描所有视频文件
                video_files = self._get_video_files_in_directory(input_path)
                if not video_files:
                    self.log_dialog.append(f"目录中未找到视频文件: {input_path}")
                    self.start_button.setEnabled(True)
                    self.input_button.setEnabled(True)
                    self.output_button.setEnabled(True)
                    return
                self.log_dialog.append(f"找到 {len(video_files)} 个视频文件")
            else:
                self.log_dialog.append(f"无效的视频输入路径: {input_path}")
                self.start_button.setEnabled(True)
                self.input_button.setEnabled(True)
                self.output_button.setEnabled(True)
                return

            frame_mode = self.frame_mode_combo.currentData()
            frame_interval = self.frame_interval_spin.value()
            target_frame_count = self.target_frame_count_spin.value()
            self.worker = ProcessWorker(
                mode="video_extract",
                output_dir=output_dir,
                video_paths=video_files,
                frame_interval=frame_interval,
                frame_mode=frame_mode,
                target_frame_count=target_frame_count,
                parent=self,
            )
        elif mode == "batch_rename":
            rename_dir = input_path
            if not os.path.isdir(rename_dir):
                self.log_dialog.append(f"目标目录不存在：{rename_dir}")
                self.start_button.setEnabled(True)
                self.input_button.setEnabled(True)
                return
            prefix = self.rename_prefix_edit.text().strip() or "Output"
            start_num = self.rename_start_spin.value()
            self.worker = ProcessWorker(
                mode='batch_rename',
                input_dir=rename_dir,
                rename_prefix=prefix,
                rename_start_num=start_num,
                parent=self,
            )

        self.worker.progress.connect(self.on_progress)
        self.worker.finished.connect(self.on_finished)
        self.worker.log.connect(self.on_worker_log)
        self.worker.start()

    def _resolve_selected_points(self, input_dir):
        return {os.path.normpath(path): list(points) for path, points in self.selected_points_map.items() if points}

    def _resolve_white_rois(self, input_dir):
        if not input_dir or not os.path.isdir(input_dir):
            return {}
        resolved = {}
        input_root = os.path.normpath(input_dir)
        for path, rois in self.white_roi_map.items():
            normalized_path = os.path.normpath(path)
            if not rois:
                continue
            try:
                if os.path.commonpath([normalized_path, input_root]) != input_root:
                    continue
            except ValueError:
                continue
            normalized_rois = []
            for roi in rois:
                if not roi:
                    continue
                x, y, w, h = roi
                if w <= 0 or h <= 0:
                    continue
                normalized_rois.append((int(x), int(y), int(w), int(h)))
            if normalized_rois:
                resolved[normalized_path] = normalized_rois
        return resolved

    def _resolve_merge_strokes(self, input_dir):
        if not input_dir or not os.path.isdir(input_dir):
            return {}
        resolved = {}
        input_root = os.path.normpath(input_dir)
        for path, strokes in self.merge_strokes_map.items():
            normalized_path = os.path.normpath(path)
            if not strokes:
                continue
            try:
                if os.path.commonpath([normalized_path, input_root]) != input_root:
                    continue
            except ValueError:
                continue
            resolved[normalized_path] = [list(stroke) for stroke in strokes if len(stroke) >= 2]
        return resolved

    def _resolve_split_strokes(self, input_dir):
        if not input_dir or not os.path.isdir(input_dir):
            return {}
        resolved = {}
        input_root = os.path.normpath(input_dir)
        for path, strokes in self.split_strokes_map.items():
            normalized_path = os.path.normpath(path)
            if not strokes:
                continue
            try:
                if os.path.commonpath([normalized_path, input_root]) != input_root:
                    continue
            except ValueError:
                continue
            resolved[normalized_path] = [list(stroke) for stroke in strokes if len(stroke) >= 2]
        return resolved

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
        painter = QPainter(base_pixmap)
        painter.setRenderHint(QPainter.Antialiasing, True)

        if self.mode_combo.currentData() == "object_slice":
            merge_pen = QPen(QColor(40, 160, 255), 3)
            merge_pen.setCapStyle(Qt.RoundCap)
            merge_pen.setJoinStyle(Qt.RoundJoin)
            painter.setPen(merge_pen)
            for stroke in self._current_merge_strokes():
                for start, end in zip(stroke, stroke[1:]):
                    painter.drawLine(QPoint(*start), QPoint(*end))

            split_pen = QPen(QColor(220, 60, 60), 3)
            split_pen.setCapStyle(Qt.RoundCap)
            split_pen.setJoinStyle(Qt.RoundJoin)
            painter.setPen(split_pen)
            for stroke in self._current_split_strokes():
                for start, end in zip(stroke, stroke[1:]):
                    painter.drawLine(QPoint(*start), QPoint(*end))

            if self.current_drag_stroke:
                if self.stroke_mode_combo.currentData() == "split":
                    temp_pen = QPen(QColor(255, 150, 120), 3)
                else:
                    temp_pen = QPen(QColor(255, 180, 40), 3)
                temp_pen.setCapStyle(Qt.RoundCap)
                temp_pen.setJoinStyle(Qt.RoundJoin)
                painter.setPen(temp_pen)
                for start, end in zip(self.current_drag_stroke, self.current_drag_stroke[1:]):
                    painter.drawLine(QPoint(*start), QPoint(*end))
        else:
            points = self._current_selected_points()
            if points:
                painter.setBrush(QBrush(QColor(220, 50, 50)))
                painter.setPen(QColor(220, 50, 50))
                for x, y in points:
                    painter.drawEllipse(QPoint(x, y), 4, 4)
            roi_pen = QPen(QColor(50, 220, 120), 2)
            roi_pen.setStyle(Qt.DashLine)
            painter.setPen(roi_pen)
            for roi in self._current_white_rois():
                rx, ry, rw, rh = roi
                painter.drawRect(rx, ry, rw, rh)
            if self.current_drag_roi:
                drag_pen = QPen(QColor(255, 90, 90) if self.roi_erase_enabled else QColor(255, 210, 80), 2)
                drag_pen.setStyle(Qt.DashLine)
                painter.setPen(drag_pen)
                rx, ry, rw, rh = self.current_drag_roi
                painter.drawRect(rx, ry, rw, rh)

        painter.end()

        scaled = base_pixmap.scaled(target_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        scaled_w = scaled.width()
        scaled_h = scaled.height()
        x_offset = max(0, (target_size.width() - scaled_w) // 2)
        y_offset = max(0, (target_size.height() - scaled_h) // 2)
        self.preview_display_rect = QRect(x_offset, y_offset, scaled_w, scaled_h)

        display_pixmap = QPixmap(scaled)
        if self.mode_combo.currentData() == "object_slice":
            action_text = "连接物件" if self.stroke_mode_combo.currentData() == "merge" else "切开物件"
            if self.stroke_mode_enabled:
                overlay_text = f"划线模式：已开启（{action_text}）"
                self.preview_label.setToolTip(f"划线模式已开启：{action_text}")
                self.preview_label.setStatusTip(f"划线模式已开启：{action_text}")
            else:
                overlay_text = f"划线模式：未开启（当前：{action_text}）"
                self.preview_label.setToolTip("")
                self.preview_label.setStatusTip("")

            overlay_painter = QPainter(display_pixmap)
            overlay_painter.setRenderHint(QPainter.Antialiasing, True)
            metrics = overlay_painter.fontMetrics()
            text_width = min(max(180, metrics.horizontalAdvance(overlay_text) + 20), max(40, display_pixmap.width() - 20))
            text_rect = QRect(10, 10, text_width, 28)
            overlay_painter.fillRect(text_rect, QColor(0, 0, 0, 160))
            overlay_painter.setPen(QColor(255, 255, 255))
            overlay_painter.drawText(text_rect.adjusted(8, 0, -8, 0), Qt.AlignLeft | Qt.AlignVCenter, overlay_text)
            overlay_painter.end()
        elif self.mode_combo.currentData() == "batch_process":
            roi_text = "已开启" if self.roi_mode_enabled else "未开启"
            erase_text = "擦除" if self.roi_erase_enabled else "新增"
            remove_text = "启用" if self.roi_remove_white_checkbox.isChecked() else "关闭"
            overlay_text = f"区域去白:{remove_text} 框选:{roi_text}/{erase_text} 选区:{len(self._current_white_rois())} 最小白块:{self.white_block_side_spin.value()}x{self.white_block_side_spin.value()}"
            overlay_painter = QPainter(display_pixmap)
            overlay_painter.setRenderHint(QPainter.Antialiasing, True)
            metrics = overlay_painter.fontMetrics()
            text_width = min(max(260, metrics.horizontalAdvance(overlay_text) + 20), max(40, display_pixmap.width() - 20))
            text_rect = QRect(10, 10, text_width, 28)
            overlay_painter.fillRect(text_rect, QColor(0, 0, 0, 160))
            overlay_painter.setPen(QColor(255, 255, 255))
            overlay_painter.drawText(text_rect.adjusted(8, 0, -8, 0), Qt.AlignLeft | Qt.AlignVCenter, overlay_text)
            overlay_painter.end()
            self.preview_label.setToolTip("")
            self.preview_label.setStatusTip("")
        else:
            self.preview_label.setToolTip("")
            self.preview_label.setStatusTip("")

        self.preview_label.setPixmap(display_pixmap)
        self.preview_label.setText("")

    def _get_latest_image_in_directory(self, dir_path):
        if not dir_path or not os.path.isdir(dir_path):
            return None
        image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp')
        latest_path = None
        latest_mtime = None
        try:
            for root, dirs, files in os.walk(dir_path):
                dirs.sort()
                for name in files:
                    if not name.lower().endswith(image_extensions):
                        continue
                    full_path = os.path.join(root, name)
                    if not os.path.isfile(full_path):
                        continue
                    mtime = os.path.getmtime(full_path)
                    if latest_path is None or mtime > latest_mtime:
                        latest_path = full_path
                        latest_mtime = mtime
        except Exception:
            return None
        return latest_path

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
            self.current_input_key = os.path.normpath(image_path)
            if self.current_input_key not in self.selected_points_map:
                self.selected_points_map[self.current_input_key] = []
            if self.current_input_key not in self.merge_strokes_map:
                self.merge_strokes_map[self.current_input_key] = []
            if self.current_input_key not in self.split_strokes_map:
                self.split_strokes_map[self.current_input_key] = []
            if self.current_input_key not in self.white_roi_map:
                self.white_roi_map[self.current_input_key] = []
            self.current_drag_stroke = None
            self.is_drawing_stroke = False
            self.roi_erase_enabled = False
            self.toggle_roi_erase_button.setChecked(False)
            self.roi_drag_start = None
            self.current_drag_roi = None
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
        target_name = os.path.basename(output_path.rstrip("\\/")) if output_path else ""
        message = f"{current}/{total} {status}: {os.path.basename(input_path)} -> {target_name}"
        if self.log_dialog:
            self.log_dialog.append(message)

        # 视频抽帧和批量重命名模式下不更新预览
        mode = self.mode_combo.currentData()
        if mode in ("video_extract", "batch_rename"):
            return
        if mode == "object_slice":
            if os.path.isfile(input_path):
                self._show_preview(input_path, update_key=True)
            return

        preview_path = input_path
        update_key = True
        if ok:
            output_candidate = None
            if output_path and os.path.isfile(output_path):
                output_candidate = output_path
            elif mode == "object_slice" and output_path and os.path.isdir(output_path):
                output_candidate = self._get_latest_image_in_directory(output_path)
            else:
                output_dir = self.output_edit.text().strip()
                if output_dir and os.path.isdir(output_dir):
                    output_candidate = self._get_latest_image_in_directory(output_dir)
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
        self.progress_bar.setValue(100)
        self.start_button.setEnabled(True)
        self.input_button.setEnabled(True)
        self.output_button.setEnabled(True)
        if self.image_button.isVisible():
            self.image_button.setEnabled(True)
        if self.clear_points_button.isVisible():
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
