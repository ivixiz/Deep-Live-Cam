# main_pyqt.py
import os
import time
import json
import shutil
import subprocess
import threading
import platform
import pyvirtualcam
from typing import Tuple, Callable
from PyQt6.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QCheckBox, QSlider,
    QComboBox, QDialog, QVBoxLayout, QHBoxLayout, QFileDialog, QScrollArea,
    QWidgetItem, QSizePolicy, QGridLayout, QLineEdit, QSpinBox, QFormLayout
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal , QThread
from PyQt6.QtGui import QPixmap, QImage

import cv2
from PIL import Image, ImageOps, ImageQt

# Ваши модули (используются как в оригинале)
import modules.globals
import modules.metadata
from modules.face_analyser import (
    get_one_face,
    get_unique_faces_from_target_image,
    get_unique_faces_from_target_video,
    add_blank_map,
    has_valid_map,
    simplify_maps,
)
from modules.capturer import get_video_frame, get_video_frame_total
from modules.processors.frame.core import get_frame_processors_modules
from modules.utilities import is_image, is_video, has_image_extension
from modules.video_capture import VideoCapturer
from modules.gettext import LanguageManager

if platform.system() == "Windows":
    try:
        from pygrabber.dshow_graph import FilterGraph
    except Exception:
        FilterGraph = None
else:
    FilterGraph = None


ROOT_WIDTH = 400
ROOT_HEIGHT = 400

PREVIEW_MAX_HEIGHT = 600
PREVIEW_MAX_WIDTH = 600
PREVIEW_DEFAULT_WIDTH = 960
PREVIEW_DEFAULT_HEIGHT = 540

MAPPER_PREVIEW_MAX_HEIGHT = 100
MAPPER_PREVIEW_MAX_WIDTH = 100

DEFAULT_BUTTON_WIDTH = 200
DEFAULT_BUTTON_HEIGHT = 40

RECENT_SOURCE = None
RECENT_TARGET = None
RECENT_OUTPUT = None
RECENT_MODEL  = modules.globals.DEFAULT_MODEL_NAME

STATUS_LABEL: QLabel | None = None 

img_ft, vid_ft = modules.globals.file_types
import modules.core as core

def save_switch_states():
    switch_states = {
        "keep_fps": modules.globals.keep_fps,
        "keep_audio": modules.globals.keep_audio,
        "keep_frames": modules.globals.keep_frames,
        "many_faces": modules.globals.many_faces,
        "map_faces": modules.globals.map_faces,
        "poisson_blend": modules.globals.poisson_blend,
        "color_correction": modules.globals.color_correction,
        "nsfw_filter": modules.globals.nsfw_filter,
        "live_mirror": modules.globals.live_mirror,
        "live_resizable": modules.globals.live_resizable,
        "fp_ui": modules.globals.fp_ui,
        "show_fps": modules.globals.show_fps,
        "mouth_mask": modules.globals.mouth_mask,
        "show_mouth_mask_box": modules.globals.show_mouth_mask_box,
        "opacity": getattr(modules.globals, "opacity", 1.0),
        "sharpness": getattr(modules.globals, "sharpness", 0.0),

        # ------------------ Virtual camera ------------------
        "vcam_width": getattr(modules.globals, "vcam_width", 640),
        "vcam_height": getattr(modules.globals, "vcam_height", 480),
        "vcam_fps": getattr(modules.globals, "vcam_fps", 30),
        "vcam_video_nr": getattr(modules.globals, "vcam_video_nr", 4),
        "vcam_card_label": getattr(modules.globals, "vcam_card_label", "DLC Webcam"),
        "vcam_device": getattr(modules.globals, "vcam_device", None),
    }
    with open("switch_states.json", "w") as f:
        json.dump(switch_states, f, indent=2)
def load_switch_states():
    try:
        with open("switch_states.json", "r") as f:
            switch_states = json.load(f)
        modules.globals.keep_fps = switch_states.get("keep_fps", True)
        modules.globals.keep_audio = switch_states.get("keep_audio", True)
        modules.globals.keep_frames = switch_states.get("keep_frames", False)
        modules.globals.many_faces = switch_states.get("many_faces", False)
        modules.globals.map_faces = switch_states.get("map_faces", False)
        modules.globals.poisson_blend = switch_states.get("poisson_blend", False)
        modules.globals.color_correction = switch_states.get("color_correction", False)
        modules.globals.nsfw_filter = switch_states.get("nsfw_filter", False)
        modules.globals.live_mirror = switch_states.get("live_mirror", False)
        modules.globals.live_resizable = switch_states.get("live_resizable", False)
        modules.globals.fp_ui = switch_states.get("fp_ui", {"face_enhancer": False})
        modules.globals.show_fps = switch_states.get("show_fps", False)
        modules.globals.mouth_mask = switch_states.get("mouth_mask", False)
        modules.globals.show_mouth_mask_box = switch_states.get("show_mouth_mask_box", False)
        modules.globals.opacity = switch_states.get("opacity", 1.0)
        modules.globals.sharpness = switch_states.get("sharpness", 0.0)

        # ------------------ Virtual camera ------------------
        modules.globals.vcam_width = switch_states.get("vcam_width", 640)
        modules.globals.vcam_height = switch_states.get("vcam_height", 480)
        modules.globals.vcam_fps = switch_states.get("vcam_fps", 30)
        modules.globals.vcam_video_nr = switch_states.get("vcam_video_nr", 4)
        modules.globals.vcam_card_label = switch_states.get("vcam_card_label", "DLC Webcam")
        modules.globals.vcam_device = switch_states.get("vcam_device", None)

    except FileNotFoundError:
        pass
def fit_image_to_size(image, width: int = None, height: int = None):
    if image is None:
        raise ValueError("Image is None")
    h, w = image.shape[:2]
    if width is None and height is None:
        return image
    ratio_w = width / w if width else None
    ratio_h = height / h if height else None
    if ratio_w is not None and ratio_h is not None:
        ratio = min(ratio_w, ratio_h)
    else:
        ratio = ratio_w if ratio_w is not None else ratio_h
    new_w = max(1, int(w * ratio))
    new_h = max(1, int(h * ratio))
    return cv2.resize(image, (new_w, new_h))
def render_preview(path: str,size: Tuple[int, int],frame_number: int = 0,) -> QPixmap:
    if not path or not os.path.isfile(path):
        return QPixmap()
    # ---------- LOAD ----------
    if is_image(path):
        frame = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if frame is None:
            return QPixmap()
    elif is_video(path):
        cap = cv2.VideoCapture(path)
        if frame_number:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ok, frame = cap.read()
        cap.release()
        if not ok or frame is None:
            return QPixmap()
    else:
        return QPixmap()
    # ---------- NORMALIZE FORMAT ----------
    if frame.ndim == 2:# GRAY → BGR
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        fmt = QImage.Format.Format_BGR888
    elif frame.shape[2] == 4:# BGRA → RGBA
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGBA)
        fmt = QImage.Format.Format_RGBA8888
    else:# BGR
        fmt = QImage.Format.Format_BGR888
    h, w, ch = frame.shape
    bytes_per_line = ch * w
    qimg = QImage(frame.data, w,h,bytes_per_line,fmt)
    if size:
        qimg = qimg.scaled(size[0],size[1],
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
    return QPixmap.fromImage(qimg)
def get_available_cameras():
    if platform.system() == "Windows" and FilterGraph is not None:
        try:
            graph = FilterGraph()
            devices = graph.get_input_devices()
            camera_indices = list(range(len(devices)))
            camera_names = devices
            if not camera_names:
                # fallback через OpenCV
                test_indices = [0, 1]
                working = []
                for idx in test_indices:
                    cap = cv2.VideoCapture(idx)
                    if cap.isOpened():
                        working.append(f"Camera {idx}")
                        cap.release()
                if working:
                    return test_indices[:len(working)], working
            if not camera_names:
                return [], ["No cameras found"]
            return camera_indices, camera_names
        except Exception:
            return [], ["No cameras found"]
    elif platform.system() == "Linux":
        camera_indices = []
        camera_names = []
        devs = []
        try:
            devs = sorted(os.listdir("/dev"))
        except Exception:
            devs = []
        for dev in devs:
            if dev.startswith("video"):
                try:
                    idx = int(dev.replace("video", ""))
                except ValueError:
                    continue
                cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
                if cap.isOpened():
                    camera_indices.append(idx)
                    camera_names.append(f"/dev/{dev}")
                    cap.release()
        if not camera_names:
            return [], ["No cameras found"]
        return camera_indices, camera_names
    else:
        # macOS or generic: try indices 0..4
        camera_indices = []
        camera_names = []
        for i in range(5):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                camera_indices.append(i)
                camera_names.append(f"Camera {i}")
                cap.release()
        if not camera_names:
            return [], ["No cameras found"]
        return camera_indices, camera_names
def connect_status_label(label: QLabel):
    global STATUS_LABEL
    STATUS_LABEL = label
def update_status(message: str):
    if STATUS_LABEL:
        STATUS_LABEL.setText(message)
    else:
        print(message) 
class ClickableLabel(QLabel):
    clicked = pyqtSignal()

    def mousePressEvent(self, event):
        self.clicked.emit()
        super().mousePressEvent(event)
class MainWindow(QWidget):
    def __init__(self, start: Callable[[], None], destroy: Callable[[], None], lang: str):
        super().__init__()

        self.lang_manager = LanguageManager(lang)
        self._ = self.lang_manager._
        load_switch_states()

        self.start_cb = start
        self.destroy_cb = destroy
        self.virtual_cam = None
        self.webcam_preview_dialog = None
        self.live_running = False

        self.setWindowTitle(f"{modules.metadata.name} {modules.metadata.version} {modules.metadata.edition}")
        self.setMinimumSize(ROOT_WIDTH, ROOT_HEIGHT)

        # main layout
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)
        
        self.source_label = self.create_clickable_label("Select Source", self.select_source_path)
        self.target_label = self.create_clickable_label("Select Target", self.select_target_path)

        labels_row = QHBoxLayout() 
        self.swap_faces_btn = QPushButton("↔")
        self.swap_faces_btn.clicked.connect(self.swap_faces_paths)
        self.swap_faces_btn.setFixedSize(20, 200) 
        labels_row.addWidget(self.source_label)
        labels_row.addWidget(self.swap_faces_btn)
        labels_row.addWidget(self.target_label)
        self.layout.addLayout(labels_row) 
    
        # ---------- switches ----------
        self.keep_fps_cb = self.make_checkbox("keep_fps", "Keep fps", "global")
        self.keep_frames_cb = self.make_checkbox("keep_frames", "Keep frames", "global")
        self.enhancer_cb = self.make_checkbox("face_enhancer", "Face Enhancer", "tumbler")
        self.keep_audio_cb = self.make_checkbox("keep_audio", "Keep audio", "global")
        self.many_faces_cb = self.make_checkbox("many_faces", "Many faces", "global")
        self.color_correction_cb = self.make_checkbox("color_correction", "Fix Blueish Cam", "global")
        self.map_faces_cb = self.make_checkbox("map_faces", "Map faces", "mapper")
        self.poisson_blend_cb = self.make_checkbox("poisson_blend", "Poisson Blend", "global")
        self.show_fps_cb = self.make_checkbox("show_fps", "Show FPS", "global")
        self.mouth_mask_cb = self.make_checkbox("mouth_mask", "Mouth Mask", "global")
        self.show_mouth_mask_box_cb = self.make_checkbox("show_mouth_mask_box", "Show Mouth Mask Box", "global")
        self.live_mirror = self.make_checkbox("live_mirror", "Mirror", "global")

        # ---------- add to grid ----------
        checkboxes = [
            self.keep_fps_cb, self.keep_frames_cb, self.enhancer_cb,
            self.keep_audio_cb, self.many_faces_cb, self.color_correction_cb,
            self.map_faces_cb, self.poisson_blend_cb, self.show_fps_cb,
            self.mouth_mask_cb, self.show_mouth_mask_box_cb, self.live_mirror
        ]

        grid = QGridLayout()
        grid.setSpacing(5)  
        grid.setContentsMargins(0, 0, 0, 0)  

        cols = 3  
        for index, cb in enumerate(checkboxes):
            row = index // cols
            col = index % cols
            grid.addWidget(cb, row, col)

        self.layout.addLayout(grid)

        left_bot_col = QVBoxLayout()
        inputdev_row = QHBoxLayout()
        self.inputdev_lab = QLabel("Input Devices:")
        cam_indices, cam_names = get_available_cameras() 
        self.cam_indices = cam_indices 
        self.cam_combo = QComboBox() 
        self.cam_combo.addItems(cam_names) 
        inputdev_row.addWidget(self.inputdev_lab)
        inputdev_row.addWidget(self.cam_combo)
        left_bot_col.addLayout(inputdev_row)

        outputdev_row = QHBoxLayout()
        self.outputdev_lab = QLabel("Output Device:")
        self.addoutdevbutt = QPushButton("Configure")
        self.addoutdevbutt.clicked.connect(self.create_virtual_camera)
        outputdev_row.addWidget(self.outputdev_lab)
        outputdev_row.addWidget(self.addoutdevbutt)
        left_bot_col.addLayout(outputdev_row)

        left_bot_col.addWidget(QLabel("Transparency"))
        self.transparency_slider = QSlider(Qt.Orientation.Horizontal)
        self.transparency_slider.setRange(0, 100)
        self.transparency_slider.setValue(int(getattr(modules.globals, "opacity", 1.0) * 100))
        self.transparency_slider.valueChanged.connect(self._on_transparency_change)
        left_bot_col.addWidget(self.transparency_slider)

        left_bot_col.addWidget(QLabel("Sharpness"))
        self.sharpness_slider = QSlider(Qt.Orientation.Horizontal)
        self.sharpness_slider.setRange(0, 50)  # *0.1 mapping
        self.sharpness_slider.setValue(int(getattr(modules.globals, "sharpness", 0.0) * 10))
        self.sharpness_slider.valueChanged.connect(self._on_sharpness_change)
        left_bot_col.addWidget(self.sharpness_slider)

        buttons_col = QVBoxLayout()

        button_width = 120
        button_height = 40

        self.live_btn = QPushButton(self._("Live"))
        self.live_btn.clicked.connect(self.open_webcam_preview)
        self.live_btn.setFixedSize(button_width, button_height)
        buttons_col.addWidget(self.live_btn)

        self.start_btn = QPushButton(self._("Render Target"))
        self.start_btn.clicked.connect(self._on_render_target)
        self.start_btn.setFixedSize(button_width, button_height)
        buttons_col.addWidget(self.start_btn)

        self.preview_btn = QPushButton(self._("Preview"))
        self.preview_btn.clicked.connect(self.toggle_preview)
        self.preview_btn.setFixedSize(button_width, button_height)
        buttons_col.addWidget(self.preview_btn)

        self.stop_btn = QPushButton(self._("Close All"))
        self.stop_btn.clicked.connect(lambda: self.destroy_cb())
        self.stop_btn.setFixedSize(button_width, button_height)
        buttons_col.addWidget(self.stop_btn)


        # status and link
        self.status_label = QLabel("")
        connect_status_label(self.status_label)
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        model_path_layout = QHBoxLayout()
        self.model_path_edit = QLineEdit()
        self.model_path_edit.setText(getattr(modules.globals, "model_path", modules.globals.DEFAULT_MODEL_NAME))
        self.model_path_edit.setPlaceholderText("inswapper_128.onnx")
        model_path_layout.addWidget(self.model_path_edit)
        self.model_path_btn = QPushButton("Browse")
        self.model_path_btn.setFixedWidth(70)
        self.model_path_btn.clicked.connect(self.select_model_path)
        model_path_layout.addWidget(self.model_path_btn)
        left_bot_col.addLayout(model_path_layout)
        left_bot_col.addWidget(self.status_label)

        controls_row = QHBoxLayout()
        controls_row.addLayout(left_bot_col)
        controls_row.addLayout(buttons_col)
        self.layout.addLayout(controls_row)

        # donate_label = QLabel(f'<a href="https://deeplivecam.net">Deep Live Cam</a>')
        # donate_label.setOpenExternalLinks(True)
        # donate_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        # self.layout.addWidget(donate_label)
        # preview dialog (hidden)
        self.preview_dialog = QDialog(self)
        self.preview_dialog.setWindowTitle(self._("Preview"))
        self.preview_dialog_layout = QVBoxLayout()
        self.preview_dialog.setLayout(self.preview_dialog_layout)
        self.preview_image_label = QLabel()
        self.preview_image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_dialog_layout.addWidget(self.preview_image_label)
        self.preview_slider = QSlider(Qt.Orientation.Horizontal)
        self.preview_slider.setRange(0, 0)
        self.preview_slider.valueChanged.connect(self.update_preview)
        # We'll show slider only for videos
        # self.preview_dialog_layout.addWidget(self.preview_slider)
        # variables for webcam preview dialog
        self.webcam_preview_dialog = None
    def create_clickable_label(self, text: str, click_callback: Callable) -> ClickableLabel:
        label = ClickableLabel(text)
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label.setFixedSize(200, 200)
        label.setScaledContents(True)
        label.setStyleSheet("font-size: 16pt; font-weight: bold; border: 1px solid gray;")
        label.clicked.connect(click_callback)
        return label
    def reset_clickable_label(self, label: ClickableLabel, text: str):
        label.setText(text)
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label.setFixedSize(200, 200)
    def create_virtual_camera(self):
        dlg = QDialog(self)
        dlg.setWindowTitle(self._("Configure Virtual Camera"))
        dlg.setModal(True)
        layout = QVBoxLayout()
        dlg.setLayout(layout)
        form = QFormLayout()
        # width/height/fps
        self.width_spin        = QSpinBox();  self.width_spin.setRange(1, 8192);   self.width_spin.setValue(getattr(modules.globals, "vcam_width", 640))
        self.height_spin       = QSpinBox();  self.height_spin.setRange(1, 8192);  self.height_spin.setValue(getattr(modules.globals, "vcam_height", 480))
        self.fps_spin          = QSpinBox();  self.fps_spin.setRange(1, 240);      self.fps_spin.setValue(getattr(modules.globals, "vcam_fps", 30))
        self.device_spin      = QSpinBox();  self.device_spin.setRange(1, 16);   self.device_spin.setValue(1)
        self.video_nr_spin     = QSpinBox();  self.video_nr_spin.setRange(0, 63);  self.video_nr_spin.setValue(getattr(modules.globals, "vcam_video_nr", 4))
        self.card_label_edit   = QLineEdit(); self.card_label_edit.setText(getattr(modules.globals, "vcam_card_label", "DLC Webcam"))
        exclusive_caps_cb = QComboBox(); exclusive_caps_cb.addItems(["0", "1"]); exclusive_caps_cb.setCurrentIndex(0)
        max_buffers_spin  = QSpinBox();  max_buffers_spin.setRange(1, 16);       max_buffers_spin.setValue(2)
        form.addRow(QLabel("Width:"), self.width_spin)
        form.addRow(QLabel("Height:"), self.height_spin)
        form.addRow(QLabel("FPS:"), self.fps_spin)
        form.addRow(QLabel("devices:"), self.device_spin)
        form.addRow(QLabel("video_nr (/dev/video4 <- 4):"), self.video_nr_spin)
        form.addRow(QLabel("card_label:"), self.card_label_edit)
        form.addRow(QLabel("exclusive_caps (0/1):"), exclusive_caps_cb)
        form.addRow(QLabel("max_buffers:"), max_buffers_spin)
        layout.addLayout(form)
  
        def set_status(msg: str, error: bool = False):
            status_label.setText(msg)
            if error:
                status_label.setStyleSheet("color: red;")
            else:
                status_label.setStyleSheet("color: gray;")
        self.debounce_timer = QTimer()
        self.debounce_timer.setSingleShot(True)
        def save_vcam_settings():
            modules.globals.vcam_width      = self.width_spin.value()
            modules.globals.vcam_height     = self.height_spin.value()
            modules.globals.vcam_fps        = self.fps_spin.value()
            modules.globals.vcam_video_nr   = self.video_nr_spin.value()
            modules.globals.vcam_card_label = self.card_label_edit.text().strip()
            modules.globals.vcam_device     = self.device_spin.value()
            save_switch_states()
            set_status("Parameters saved", False)
        self.debounce_timer.timeout.connect(save_vcam_settings)
        def debounce(*args):
            set_status("debounce...",False)
            self.debounce_timer.start(1000)  
        self.width_spin.valueChanged.connect(debounce)
        self.height_spin.valueChanged.connect(debounce)
        self.fps_spin.valueChanged.connect(debounce)
        self.device_spin.valueChanged.connect(debounce)
        self.video_nr_spin.valueChanged.connect(debounce)
        self.card_label_edit.textChanged.connect(debounce)
        
        width = self.width_spin.value()
        height = self.height_spin.value()
        fps = self.fps_spin.value()
        device = self.device_spin.value()
        video_nr = self.video_nr_spin.value()
        card_label = self.card_label_edit.text().strip()
        exclusive_caps = exclusive_caps_cb.currentText()
        max_buffers = max_buffers_spin.value()

        status_label = QLabel("")
        status_label.setWordWrap(True)
        layout.addWidget(status_label)
        btn_row = QHBoxLayout()
        apply_btn = QPushButton(self._("Apply (load module)"))
        test_btn = QPushButton(self._("Test / Check device"))
        cancel_btn = QPushButton(self._("Close"))
        btn_row.addWidget(apply_btn)
        btn_row.addWidget(test_btn)
        btn_row.addWidget(cancel_btn)
        layout.addLayout(btn_row)
        
        def device_path(n: int) -> str:
            return f"/dev/video{n}"           
        def check_device_exists(n: int) -> bool:
            return os.path.exists(device_path(n))
        def build_modprobe_command(devices, video_nr, card_label, exclusive_caps, max_buffers):
            cmd = (
                f"modprobe v4l2loopback devices={devices} video_nr={video_nr} "
                f"card_label=\"{card_label}\" exclusive_caps={exclusive_caps} max_buffers={max_buffers}"
            )
            return cmd
        def run_command_as_root(cmd_shell: str) -> tuple[int, str]:
            # Try pkexec if present
            pkexec_path = shutil.which("pkexec")
            try_cmds = []
            if pkexec_path:
                # pkexec runs a program; run sh -c "cmd"
                try_cmds.append([pkexec_path, "bash", "-c", cmd_shell])
            # fallback to sudo (note: this will prompt in terminal)
            sudo_path = shutil.which("sudo")
            if sudo_path:
                try_cmds.append([sudo_path, "bash", "-c", cmd_shell])

            if not try_cmds:
                return (1, "No pkexec or sudo found on PATH; cannot escalate privileges from GUI.")

            last_out = ""
            last_code = 1
            for cmd in try_cmds:
                try:
                    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                    last_code = proc.returncode
                    last_out = proc.stdout
                    # If pkexec returns 0, success — don't try sudo
                    if last_code == 0:
                        break
                except Exception as e:
                    last_out = str(e)
                    last_code = 1
            return last_code, last_out
        def apply_clicked():
            if not card_label:
                set_status("card_label cannot be empty")
                return
            # build shell command: rmmod (ignore errors) then modprobe
            modprobe_cmd = build_modprobe_command(device, video_nr, card_label, exclusive_caps, max_buffers)
            full_cmd = f"set -e; rmmod v4l2loopback || true; {modprobe_cmd}"
            set_status(self._("Running commands, waiting for privilege prompt..."))
            def worker():
                rc, out = run_command_as_root(full_cmd)
                if rc == 0:
                    time.sleep(0.3)
                    exists = check_device_exists(video_nr)
                    if exists:
                        def update_success():
                            self.vcam_width = width
                            self.vcam_height = height
                            self.vcam_fps = fps
                            self.vcam_video_nr = video_nr
                            self.vcam_device = device_path(video_nr)
                            self.vcam_card_label = card_label
                            self.addoutdevbutt.setText(self.vcam_device)
                            self.save_vcam_settings() 
                            set_status(self._("v4l2loopback loaded successfully: ") + self.vcam_device)
                            
                        self.run_on_ui_thread(update_success)
                    else:
                        def update_no_device():
                            set_status(self._("Module loaded but device not found: ") + device_path(video_nr), True)
                        self.run_on_ui_thread(update_no_device)
                else:
                    def update_fail():
                        set_status(self._("Command failed: ") + out.splitlines()[-10:], True)
                    self.run_on_ui_thread(update_fail)
            threading.Thread(target=worker, daemon=True).start()
        def test_clicked():
            video_nr = self.video_nr_spin.value()
            if check_device_exists(video_nr):
                set_status(self._("Device exists: ") + device_path(video_nr))
            else:
                set_status(self._("Device not found: ") + device_path(video_nr), True)
        def close_clicked():
            if check_device_exists(video_nr):
                self.addoutdevbutt.setText(device_path(video_nr))
            dlg.close()    
        def run_on_ui_thread(fn):
            QTimer.singleShot(0, fn)
        self.run_on_ui_thread = run_on_ui_thread
        apply_btn.clicked.connect(apply_clicked)
        test_btn.clicked.connect(test_clicked)
        cancel_btn.clicked.connect(close_clicked)
        dlg.exec()
    def make_checkbox(self, key: str, text: str, toggle_type: str = "global"):
        cb = QCheckBox(self._(text))
        # initial state
        if toggle_type == "tumbler":
            cb.setChecked(modules.globals.fp_ui.get(key, False))
            cb.stateChanged.connect(lambda state: self.update_tumbler(key,cb.isChecked()))
        elif toggle_type == "global":
            cb.setChecked(getattr(modules.globals, key, False))
            cb.stateChanged.connect(lambda _: self.update_global(key, cb.isChecked()))
        elif toggle_type == "mapper":
            cb.setChecked(getattr(modules.globals, key, False))
            cb.stateChanged.connect(lambda _: self.toggle_mapper(cb.isChecked()))
        return cb
    def update_global(self, name, value):
        setattr(modules.globals, name, value)
        save_switch_states()
    def update_tumbler(self, key, value):
        modules.globals.fp_ui[key] = value
        save_switch_states()
        if hasattr(self, "webcam_preview_dialog") and \
        self.webcam_preview_dialog is not None and \
        self.webcam_preview_dialog.isVisible():
            global frame_processors
            frame_processors = get_frame_processors_modules(modules.globals.frame_processors)
        # update frame processors if preview open (best-effort)
        # if self.preview_dialog.isVisible(): ... (not required here)
    def toggle_mapper(self, enabled: bool):
        modules.globals.map_faces = enabled
        save_switch_states()
    def _on_transparency_change(self, value):
        val = value / 100.0
        modules.globals.opacity = val
        save_switch_states()
        if val == 0:
            modules.globals.fp_ui["face_enhancer"] = False
            self.status_label.setText("Transparency set to 0% - Face swapping disabled.")
        elif val == 1.0:
            modules.globals.face_swapper_enabled = True
            self.status_label.setText("Transparency set to 100%.")
        else:
            modules.globals.face_swapper_enabled = True
            self.status_label.setText(f"Transparency set to {int(val * 100)}%")
    def _on_sharpness_change(self, value):
        val = value / 10.0
        modules.globals.sharpness = val
        save_switch_states()
        self.status_label.setText(f"Sharpness set to {val:.1f}")    
    def _on_render_target(self):
        if not self.select_output_path():
            return
        if self.start_cb is None:
            print("start_cb is None! Cannot call start function.")
            return
        self.start_cb()
    def select_source_path(self):
        global RECENT_SOURCE
        file_filter = "Images (*.png *.jpg *.jpeg *.bmp *.gif *.tiff)"
        source_path, _ = QFileDialog.getOpenFileName(self, self._("Select a source image"), 
                                                     RECENT_SOURCE or modules.globals.resources_dir, 
                                                     file_filter)
        print(f"source_path = {source_path}")
        if source_path and os.path.isfile(source_path):
            RECENT_SOURCE = source_path
            modules.globals.source_path = source_path
            pixmap = render_preview(source_path, (200, 200))
            self.source_label.setPixmap(pixmap)
            modules.globals.source_face_update = True
            modules.globals.FACE_SWAPPER = None 
        else:
            modules.globals.source_path = None
            self.reset_clickable_label(self.source_label, "Select Source")
    def select_output_path(self):
        global RECENT_OUTPUT
        output_path, _ = QFileDialog.getSaveFileName(
            self,
            self._("Select output file"),
            RECENT_OUTPUT or modules.globals.resources_dir ,
        )
        print(f"output_path = {output_path}")
        if not output_path:
            modules.globals.output_path = None
            return False
        RECENT_OUTPUT = output_path
        modules.globals.output_path = output_path
        self.status_label.setText(f"Output: {output_path}")
        return True
    def select_target_path(self):
        global RECENT_TARGET
        # Filter includes images and common videos
        file_filter = "Media (*.png *.jpg *.jpeg *.bmp *.mp4 *.mov *.avi *.mkv)"
        target_path, _ = QFileDialog.getOpenFileName(self, self._("Select a target image or video"), 
                                              RECENT_TARGET or  modules.globals.resources_dir,
                                              file_filter)
        if target_path and os.path.isfile(target_path):
            RECENT_TARGET = target_path
            modules.globals.target_path = target_path
            print(f"target_path = {target_path}")
            if is_image(target_path):
                self.target_label.setPixmap(render_preview(target_path, (200,200)))
            elif is_video(target_path):
                pix = render_preview(target_path, (200,200))
                self.target_label.setPixmap(pix)
            else:
                modules.globals.target_path = None
                self.target_label.clear()
        else:
            modules.globals.target_path = None
            self.reset_clickable_label(self.target_label, "Select Target")
    def select_model_path(self):
        global RECENT_MODEL
        file_filter = "ONNX Models (*.onnx);;All Files (*)"
        model_path, _ = QFileDialog.getOpenFileName(self, "Select Face Swapper Model",  
                                               modules.globals.models_dir or RECENT_MODEL,
                                               file_filter)
        if model_path:
            RECENT_MODEL = os.path.dirname(model_path)
            modules.globals.model_path = model_path
            modules.globals.FACE_SWAPPER = None 
            self.model_path_edit.setText(model_path)
            self.status_label.setText(f"model_path = {model_path}")
    def swap_faces_paths(self):
        global RECENT_DIRECTORY_SOURCE, RECENT_DIRECTORY_TARGET
        source_path = modules.globals.source_path
        target_path = modules.globals.target_path
        if not is_image(source_path) or not is_image(target_path):
            return
        modules.globals.source_path = target_path
        modules.globals.target_path = source_path
        RECENT_DIRECTORY_SOURCE = os.path.dirname(modules.globals.source_path)
        RECENT_DIRECTORY_TARGET = os.path.dirname(modules.globals.target_path)
        # update previews
        self.source_label.setPixmap(render_preview(modules.globals.source_path, (200,200)))
        self.target_label.setPixmap(render_preview(modules.globals.target_path, (200,200)))
    def toggle_preview(self):
        if not modules.globals.source_path or not modules.globals.target_path:
            self.status_label.setText("Select source and target first")
            return
        if self.preview_dialog.isVisible():
            self.preview_dialog.hide()
        elif modules.globals.source_path and modules.globals.target_path:
            self.init_preview()
            self.update_preview()    
    def init_preview(self):
        # if video target, configure slider
        self.status_label.setText("Processing...")
        if is_image(modules.globals.target_path):
            if self.preview_slider.parent() is not None:
                self.preview_dialog_layout.removeWidget(self.preview_slider)
                self.preview_slider.hide()
        elif is_video(modules.globals.target_path):
            total = get_video_frame_total(modules.globals.target_path)
            self.preview_slider.setRange(0, total if total>0 else 0)
            if self.preview_slider.parent() is None or not self.preview_slider.isVisible():
                self.preview_dialog_layout.addWidget(self.preview_slider)
                self.preview_slider.show()
            self.preview_slider.setValue(0)
    def update_preview(self, value=0):
        if not (modules.globals.source_path and modules.globals.target_path):
            return
        self.status_label.setText("Processing...")
        frame_num = int(value)
        temp_frame = get_video_frame(modules.globals.target_path, frame_num)
        if temp_frame is None:
            return
        src_face = get_one_face(cv2.imread(modules.globals.source_path))
        for frame_processor in get_frame_processors_modules(
            modules.globals.frame_processors
        ):
            temp_frame = frame_processor.process_frame(src_face, temp_frame)
        temp_frame = fit_image_to_size(
            temp_frame, PREVIEW_MAX_WIDTH, PREVIEW_MAX_HEIGHT
        )
        h, w, ch = temp_frame.shape
        bytes_per_line = ch * w
        qimg = QImage(temp_frame.data,w,h,bytes_per_line,QImage.Format.Format_RGB888,)
        self.preview_image_label.setPixmap(QPixmap.fromImage(qimg))
        self.preview_dialog.show()
        self.preview_dialog.raise_()
        self.preview_dialog.activateWindow()
        self.status_label.setText("Processing succeed!")
    def open_webcam_preview(self):
        if not self.live_running:
            self.start_live()
        else:
            self.stop_live()
    def start_live(self):
        camera_index = self.cam_indices[self.cam_combo.currentIndex()]
        self.virtual_cam = WebcamVirtualThread(camera_index, 
                                               modules.globals.vcam_width, 
                                               modules.globals.vcam_height,
                                               modules.globals.vcam_fps)
        self.virtual_cam.start()

        self.webcam_preview_dialog = WebcamPreviewDialog(self.virtual_cam)
        self.webcam_preview_dialog.show()

        self.live_btn.setText("Live Stop")
        self.live_running = True
    def stop_live(self):
        if self.webcam_preview_dialog:
            self.webcam_preview_dialog.close()
            self.webcam_preview_dialog = None

        if self.virtual_cam:
            self.virtual_cam.stop()
            self.virtual_cam = None

        self.live_btn.setText("Live")
        self.live_running = False
class CameraWorker(QThread):

    frame_ready = pyqtSignal(object)  # Emits processed frame
    fps_updated = pyqtSignal(float)
    def __init__(self, cap: VideoCapturer, face_enhancers, other_processors, parent=None):
        super().__init__(parent)
        self.cap = cap
        self.face_enhancers = face_enhancers
        self.other_processors = other_processors
        self.running = True
        self.source_image = None
        self.frame_count = 0
        self.prev_time = time.time()
        self.fps_update_interval = 0.5
        self.fps = 0.0
    def run(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret or frame is None:
                continue

            temp_frame = frame

            if modules.globals.live_mirror:
                temp_frame = cv2.flip(temp_frame, 1)

            if not modules.globals.map_faces and modules.globals.source_path:
                if self.source_image is None or modules.globals.source_face_update:
                    img = cv2.imread(modules.globals.source_path)
                    if img is not None:
                        self.source_image = get_one_face(img)
                        modules.globals.source_face_update = False
            # -------- processing --------
            if modules.globals.map_faces:
                modules.globals.target_path = None

                for proc in self.face_enhancers:
                    if modules.globals.fp_ui.get("face_enhancer", False):
                        temp_frame = proc.process_frame_v2(temp_frame)

                for proc in self.other_processors:
                    temp_frame = proc.process_frame_v2(temp_frame)

            else:
                for proc in self.face_enhancers:
                    if modules.globals.fp_ui.get("face_enhancer", False):
                        temp_frame = proc.process_frame(None, temp_frame)

                for proc in self.other_processors:
                    temp_frame = proc.process_frame(self.source_image, temp_frame)

            self.fps_step()
            self.frame_ready.emit(temp_frame)
    def fps_step(self):
        self.frame_count += 1
        current_time = time.time()
        if current_time - self.prev_time >= self.fps_update_interval:
            fps = self.frame_count / (current_time - self.prev_time)
            self.frame_count = 0
            self.prev_time = current_time
            self.fps_updated.emit(fps)
    def stop(self):
        self.running = False
        self.wait()
class WebcamVirtualThread(QThread):
    frame_ready = pyqtSignal(object)
    fps_updated = pyqtSignal(float)
    def __init__(self, camera_index: int, vcam_width = 640, vcam_height = 480, vcam_fps = 60, vcam_device ="/dev/video4"):
        super().__init__()
        self.camera_index = camera_index
        self.running = False

        self.cap = None
        self.vcam_width = vcam_width
        self.vcam_height = vcam_height
        self.vcam_fps = vcam_fps
        self.vcam_device = vcam_device
        self.vcam = None
        self.worker = None
        try:
            print(f"vcam_width = {self.vcam_width} | vcam_height = {self.vcam_height}")
            self.vcam = pyvirtualcam.Camera(
                width=self.vcam_width, 
                height=self.vcam_height, 
                fps=self.vcam_fps, 
                device=self.vcam_device,
                fmt=pyvirtualcam.PixelFormat.BGR
            )
        except Exception as e:
            print(f"Failed to send frame to virtual cam: {e}")
    def run(self):
        self.running = True
        # --- init real camera ---
        self.cap = VideoCapturer(self.camera_index)
        if not self.cap.start(PREVIEW_DEFAULT_WIDTH, PREVIEW_DEFAULT_HEIGHT, self.vcam_fps):
            return

        processors = get_frame_processors_modules(modules.globals.frame_processors)

        self.worker = CameraWorker(self.cap, [], processors)
        self.worker.frame_ready.connect(self._on_frame)
        self.worker.fps_updated.connect(self.fps_updated)
        self.worker.start()

        self.exec()
    def _on_frame(self, frame):
        if not self.running:
            return

        out = cv2.resize(frame, (self.vcam_width, self.vcam_height))
        try:
            self.vcam.send(out)
            self.vcam.sleep_until_next_frame()
        except Exception as e:
            pass

        self.frame_ready.emit(frame)#ui signal
    def _on_fps(self, fps: float):
        self.fps = fps
        self.fps_updated.emit(fps)    
    def stop(self):
        self.running = False

        if self.worker:
            self.worker.stop()
        if self.cap:
            self.cap.release()
        if self.vcam:
            self.vcam.close()

        self.quit()
        self.wait()
class WebcamPreviewDialog(QDialog):
    def __init__(self, virtual_cam: WebcamVirtualThread):
        super().__init__()
        self.setWindowTitle("Webcam Preview")
        self.setMinimumSize(480, 480)
       
        self.label = QLabel(alignment=Qt.AlignmentFlag.AlignCenter)
        QVBoxLayout(self).addWidget(self.label)

        self.virtual_cam = virtual_cam
        self.fps = 0.0
        
        self.virtual_cam.frame_ready.connect(self.update_frame)
        self.virtual_cam.fps_updated.connect(self._on_fps)
    def _on_fps(self, fps: float):
        self.fps = fps
    def update_frame(self, frame):
        frame = fit_image_to_size(frame, self.width(), self.height())
        
        if modules.globals.show_fps:
            cv2.putText(frame, f"FPS: {self.fps:.1f}", (10, 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        h, w, ch = frame.shape
        img = QImage(frame.data, w, h, ch*w, QImage.Format.Format_BGR888)
        self.label.setPixmap(QPixmap.fromImage(img))


