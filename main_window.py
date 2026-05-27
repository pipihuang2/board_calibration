from __future__ import annotations
import cv2
import math
import numpy as np

def best_fraction(ratio: float,
                  max_num: int = 128,   # 后分配器上限
                  max_den: int = 32     # 乘法器上限
                  ) -> tuple[int, int]:
    """Return (numerator, denominator) closest to ratio within the given ranges."""
    best_num, best_den = 1, 1
    best_err = float("inf")
    for den in range(1, max_den + 1):
        for num in range(1, max_num + 1):
            err = abs(num / den - ratio)
            if err < best_err:
                best_err = err
                best_num, best_den = num, den
    return best_num, best_den
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QAction, QFont, QColor, QPalette
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QPushButton, QLabel, QTableWidget, QTableWidgetItem,
    QFileDialog, QHeaderView, QFrame, QGraphicsDropShadowEffect,
    QLineEdit, QSizePolicy, QMessageBox, QCheckBox,
)

from image_view import ImageView, RotatedRoi
from ellipse_detector import detect_ellipses

# ── Palette ────────────────────────────────────────────────────────────────
_IMG_BG      = "#EAECF0"   # image area background
_PANEL       = "#111827"   # dark navy sidebar
_CARD        = "#1F2937"   # metric card bg
_CARD2       = "#374151"   # slightly lighter card
_ACCENT      = "#3B82F6"   # blue
_ACCENT_HV   = "#60A5FA"
_ACCENT_PR   = "#2563EB"
_GREEN       = "#10B981"   # emerald — freq display
_TEXT        = "#F9FAFB"   # near-white body text
_TEXT_MUTED  = "#9CA3AF"   # slate muted
_BORDER      = "#374151"   # dark border
_BORDER_CARD = "#4B5563"   # card inner border

STYLESHEET = f"""
/* ── Base ─────────────────────────────────────────── */
QMainWindow {{ background: {_IMG_BG}; }}
QWidget#central {{ background: {_IMG_BG}; }}

/* ── Dark sidebar ─────────────────────────────────── */
QFrame#panel {{
    background: {_PANEL};
    border-radius: 0px;
}}

/* ── Section label ────────────────────────────────── */
QLabel#section_label {{
    color: {_TEXT_MUTED};
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 1.2px;
}}

/* ── Metric card ──────────────────────────────────── */
QFrame#metric_card {{
    background: {_CARD};
    border-radius: 10px;
    border: 1px solid {_BORDER};
}}

/* ── Pill button – primary ────────────────────────── */
QPushButton {{
    background: {_ACCENT};
    color: #FFFFFF;
    border: none;
    border-radius: 16px;
    font-size: 13px;
    font-weight: 600;
    min-height: 38px;
    max-height: 38px;
}}
QPushButton:hover   {{ background: {_ACCENT_HV}; }}
QPushButton:pressed {{ background: {_ACCENT_PR}; }}
QPushButton:disabled {{
    background: #374151;
    color: #6B7280;
}}

/* ── Ghost button – clear ─────────────────────────── */
QPushButton#btn_clear {{
    background: transparent;
    color: {_TEXT_MUTED};
    border: 1.5px solid {_BORDER_CARD};
    border-radius: 16px;
}}
QPushButton#btn_clear:hover {{
    color: {_TEXT};
    border-color: {_TEXT_MUTED};
    background: rgba(255,255,255,0.05);
}}
QPushButton#btn_clear:pressed {{
    background: rgba(255,255,255,0.10);
}}
QPushButton#btn_clear:disabled {{
    color: #4B5563;
    border-color: #374151;
}}

/* ── Table ────────────────────────────────────────── */
QTableWidget {{
    border: 1px solid {_BORDER};
    border-radius: 8px;
    background: {_CARD};
    color: {_TEXT};
    font-size: 12px;
    gridline-color: {_BORDER};
    selection-background-color: rgba(59,130,246,0.25);
    selection-color: {_TEXT};
    outline: none;
}}
QTableWidget::item {{ padding: 4px 0; color: {_TEXT}; }}
QTableWidget::item:alternate {{ background: #253044; }}
QHeaderView::section {{
    background: {_CARD2};
    color: {_TEXT_MUTED};
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 0.8px;
    border: none;
    border-bottom: 1px solid {_BORDER};
    padding: 6px 0;
    text-transform: uppercase;
}}
QHeaderView {{ background: transparent; }}

/* ── Scrollbar ────────────────────────────────────── */
QScrollBar:vertical {{
    background: transparent;
    width: 5px;
    margin: 0;
}}
QScrollBar::handle:vertical {{
    background: #4B5563;
    border-radius: 2px;
    min-height: 20px;
}}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}

/* ── Input ────────────────────────────────────────── */
QLineEdit {{
    border: 1.5px solid {_BORDER_CARD};
    border-radius: 10px;
    padding: 4px 12px;
    font-size: 13px;
    background: {_PANEL};
    color: {_TEXT};
    min-height: 34px;
    max-height: 34px;
}}
QLineEdit:focus {{
    border-color: {_ACCENT};
    background: {_CARD};
}}

/* ── Status bar ───────────────────────────────────── */
QStatusBar {{
    background: {_PANEL};
    color: {_TEXT_MUTED};
    font-size: 11px;
    border-top: 1px solid {_BORDER};
    padding-left: 8px;
}}

/* ── Menu bar ─────────────────────────────────────── */
QMenuBar {{
    background: {_PANEL};
    color: {_TEXT_MUTED};
    font-size: 13px;
    border-bottom: 1px solid {_BORDER};
}}
QMenuBar::item {{ padding: 4px 10px; border-radius: 4px; }}
QMenuBar::item:selected {{ background: rgba(255,255,255,0.08); color: {_TEXT}; }}
QMenu {{
    background: {_CARD};
    border: 1px solid {_BORDER_CARD};
    border-radius: 8px;
    padding: 4px;
    color: {_TEXT};
}}
QMenu::item {{ padding: 7px 20px; border-radius: 4px; }}
QMenu::item:selected {{ background: rgba(59,130,246,0.2); }}
QMenu::separator {{ height: 1px; background: {_BORDER}; margin: 3px 0; }}
"""


def _shadow(widget, radius=24, opacity=60):
    eff = QGraphicsDropShadowEffect(widget)
    eff.setBlurRadius(radius)
    eff.setOffset(-2, 0)
    eff.setColor(QColor(0, 0, 0, opacity))
    widget.setGraphicsEffect(eff)


def _section_label(text: str) -> QLabel:
    lbl = QLabel(text.upper())
    lbl.setObjectName("section_label")
    return lbl


def _make_metric_card(title: str, value_label: QLabel) -> QFrame:
    card = QFrame()
    card.setObjectName("metric_card")
    lay = QVBoxLayout(card)
    lay.setContentsMargins(14, 10, 14, 12)
    lay.setSpacing(2)
    t = QLabel(title.upper())
    t.setObjectName("section_label")
    t.setAlignment(Qt.AlignmentFlag.AlignCenter)
    lay.addWidget(t)
    value_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
    lay.addWidget(value_label)
    return card


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("线扫相机标定分析")
        self.resize(1300, 820)
        self.setStyleSheet(STYLESHEET)

        self._bgr_image: np.ndarray | None = None
        self._current_roi: RotatedRoi | None = None

        self._build_ui()
        self._build_menu()

    # ------------------------------------------------------------------ #
    def _build_ui(self):
        central = QWidget()
        central.setObjectName("central")
        self.setCentralWidget(central)

        root = QHBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ── Left: image view ──────────────────────────────────────────
        img_wrap = QWidget()
        img_wrap.setObjectName("central")
        img_lay = QVBoxLayout(img_wrap)
        img_lay.setContentsMargins(16, 14, 12, 14)

        self.image_view = ImageView()
        self.image_view.roi_selected.connect(self._on_roi_selected)
        img_lay.addWidget(self.image_view)
        root.addWidget(img_wrap, stretch=1)

        # ── Right: dark sidebar ───────────────────────────────────────
        panel = QFrame()
        panel.setObjectName("panel")
        panel.setFixedWidth(300)
        _shadow(panel)

        pl = QVBoxLayout(panel)
        pl.setContentsMargins(16, 18, 16, 16)
        pl.setSpacing(8)

        # App title area
        app_title = QLabel("标定分析")
        app_title.setStyleSheet(f"color: {_TEXT}; font-size: 16px; font-weight: 700;")
        app_sub = QLabel("线扫相机椭圆检测工具")
        app_sub.setStyleSheet(f"color: {_TEXT_MUTED}; font-size: 10px;")
        pl.addWidget(app_title)
        pl.addWidget(app_sub)
        pl.addSpacing(6)

        # ── Buttons ───────────────────────────────────────────────────
        self.btn_open = QPushButton("打开图片")
        self.btn_open.clicked.connect(self._open_image)

        self.btn_rotate_left = QPushButton("左旋转")
        self.btn_rotate_left.setEnabled(False)
        self.btn_rotate_left.clicked.connect(self._rotate_left)

        self.btn_rotate_right = QPushButton("右旋转")
        self.btn_rotate_right.setEnabled(False)
        self.btn_rotate_right.clicked.connect(self._rotate_right)

        self.btn_detect = QPushButton("检测椭圆")
        self.btn_detect.setEnabled(False)
        self.btn_detect.clicked.connect(self._detect)

        self.btn_clear = QPushButton("清除")
        self.btn_clear.setObjectName("btn_clear")
        self.btn_clear.setEnabled(False)
        self.btn_clear.clicked.connect(self._clear_roi)

        pl.addWidget(self.btn_open)

        rot_row = QHBoxLayout()
        rot_row.setSpacing(8)
        rot_row.addWidget(self.btn_rotate_left)
        rot_row.addWidget(self.btn_rotate_right)
        pl.addLayout(rot_row)

        det_row = QHBoxLayout()
        det_row.setSpacing(8)
        det_row.addWidget(self.btn_detect)
        det_row.addWidget(self.btn_clear)
        pl.addLayout(det_row)

        pl.addSpacing(6)

        # ── 参数输入 card (二值化 + 曝光 + 圆直径) ────────────────────
        input_card = QFrame()
        input_card.setObjectName("metric_card")
        in_lay = QVBoxLayout(input_card)
        in_lay.setContentsMargins(14, 10, 14, 12)
        in_lay.setSpacing(6)
        in_lay.addWidget(_section_label("参数输入"))

        thr_row = QHBoxLayout()
        thr_row.setSpacing(8)
        thr_lbl = QLabel("二值化")
        thr_lbl.setStyleSheet(f"color: {_TEXT_MUTED}; font-size: 12px;")
        thr_row.addWidget(thr_lbl)
        self.thr_auto = QCheckBox("自动")
        self.thr_auto.setChecked(True)
        self.thr_auto.setStyleSheet(f"color: {_TEXT_MUTED}; font-size: 12px;")
        thr_row.addWidget(self.thr_auto)
        self.thr_input = QLineEdit("127")
        self.thr_input.setPlaceholderText("0–255")
        self.thr_input.setEnabled(False)
        thr_row.addWidget(self.thr_input)
        self.thr_auto.toggled.connect(lambda checked: self.thr_input.setEnabled(not checked))
        in_lay.addLayout(thr_row)

        exp_row = QHBoxLayout()
        exp_row.setSpacing(8)
        exp_lbl = QLabel("曝光时间")
        exp_lbl.setStyleSheet(f"color: {_TEXT_MUTED}; font-size: 12px;")
        exp_row.addWidget(exp_lbl)
        self.exp_input = QLineEdit()
        self.exp_input.setPlaceholderText("μs")
        self.exp_input.textChanged.connect(self._on_exposure_changed)
        exp_row.addWidget(self.exp_input)
        in_lay.addLayout(exp_row)

        diam_row = QHBoxLayout()
        diam_row.setSpacing(8)
        diam_lbl = QLabel("圆直径")
        diam_lbl.setStyleSheet(f"color: {_TEXT_MUTED}; font-size: 12px;")
        diam_row.addWidget(diam_lbl)
        self.circle_diam_input = QLineEdit()
        self.circle_diam_input.setPlaceholderText("mm")
        self.circle_diam_input.textChanged.connect(self._on_scale_input_changed)
        diam_row.addWidget(self.circle_diam_input)
        in_lay.addLayout(diam_row)

        pl.addWidget(input_card)
        pl.addSpacing(4)

        # ── Image info card (compact, single row) ─────────────────────
        img_card = QFrame()
        img_card.setObjectName("metric_card")
        ic_lay = QVBoxLayout(img_card)
        ic_lay.setContentsMargins(14, 8, 14, 8)
        ic_lay.setSpacing(4)
        ic_lay.addWidget(_section_label("图像信息"))

        info_row = QHBoxLayout()
        info_row.setSpacing(6)
        x_key = QLabel("X")
        x_key.setStyleSheet(f"color: {_TEXT_MUTED}; font-size: 11px;")
        self.lbl_img_x = QLabel("—")
        self.lbl_img_x.setStyleSheet(f"color: {_TEXT}; font-size: 12px; font-weight: 600;")
        y_key = QLabel("Y")
        y_key.setStyleSheet(f"color: {_TEXT_MUTED}; font-size: 11px;")
        self.lbl_img_y = QLabel("—")
        self.lbl_img_y.setStyleSheet(f"color: {_TEXT}; font-size: 12px; font-weight: 600;")
        info_row.addStretch()
        info_row.addWidget(x_key)
        info_row.addWidget(self.lbl_img_x)
        info_row.addSpacing(20)
        info_row.addWidget(y_key)
        info_row.addWidget(self.lbl_img_y)
        info_row.addStretch()
        ic_lay.addLayout(info_row)
        pl.addWidget(img_card)

        pl.addSpacing(4)

        # ── Detection results table ───────────────────────────────────
        pl.addWidget(_section_label("检测结果"))

        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["#", "X轴(px)", "Y轴(px)", "Y/X"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setAlternatingRowColors(True)
        self.table.verticalHeader().setVisible(False)
        self.table.setShowGrid(False)
        self.table.setMinimumHeight(140)
        pl.addWidget(self.table, stretch=1)

        pl.addSpacing(4)

        # ── 测量结果 combined card ────────────────────────────────────
        result_card = QFrame()
        result_card.setObjectName("metric_card")
        rl = QVBoxLayout(result_card)
        rl.setContentsMargins(14, 10, 14, 12)
        rl.setSpacing(6)

        def _param_col(title: str, color: str = _TEXT, font_size: int = 18):
            col = QVBoxLayout()
            col.setSpacing(2)
            val_lbl = QLabel("—")
            fv = QFont(); fv.setPointSize(font_size); fv.setBold(True)
            val_lbl.setFont(fv)
            val_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            val_lbl.setStyleSheet(f"color: {color};")
            ttl_lbl = QLabel(title)
            ttl_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            ttl_lbl.setObjectName("section_label")
            col.addWidget(val_lbl)
            col.addWidget(ttl_lbl)
            return col, val_lbl

        # 平均 Y / X
        avg_hdr = QLabel("平均 Y / X 比值".upper())
        avg_hdr.setObjectName("section_label")
        avg_hdr.setAlignment(Qt.AlignmentFlag.AlignCenter)
        rl.addWidget(avg_hdr)

        self.avg_label = QLabel("—")
        f1 = QFont(); f1.setPointSize(20); f1.setBold(True)
        self.avg_label.setFont(f1)
        self.avg_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.avg_label.setStyleSheet(f"color: {_ACCENT};")
        rl.addWidget(self.avg_label)

        sep1 = QFrame()
        sep1.setFrameShape(QFrame.Shape.HLine)
        sep1.setStyleSheet(f"color: {_BORDER};")
        rl.addWidget(sep1)

        # 乘法器 | 后分配器
        cols = QHBoxLayout()
        cols.setSpacing(0)
        col_mult, self.lbl_mult     = _param_col("乘法器")
        col_post, self.lbl_post_div = _param_col("后分配器")
        vdiv = QFrame()
        vdiv.setFrameShape(QFrame.Shape.VLine)
        vdiv.setStyleSheet(f"color: {_BORDER};")
        cols.addLayout(col_mult)
        cols.addWidget(vdiv)
        cols.addLayout(col_post)
        rl.addLayout(cols)

        sep2 = QFrame()
        sep2.setFrameShape(QFrame.Shape.HLine)
        sep2.setStyleSheet(f"color: {_BORDER};")
        rl.addWidget(sep2)

        # 最大采集频率
        freq_hdr = QLabel("最大采集频率".upper())
        freq_hdr.setObjectName("section_label")
        freq_hdr.setAlignment(Qt.AlignmentFlag.AlignCenter)
        rl.addWidget(freq_hdr)

        self.freq_label = QLabel("—")
        f2 = QFont(); f2.setPointSize(18); f2.setBold(True)
        self.freq_label.setFont(f2)
        self.freq_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.freq_label.setStyleSheet(f"color: {_GREEN};")
        rl.addWidget(self.freq_label)

        sep3 = QFrame()
        sep3.setFrameShape(QFrame.Shape.HLine)
        sep3.setStyleSheet(f"color: {_BORDER};")
        rl.addWidget(sep3)

        # 比例尺
        scale_hdr = QLabel("比例尺".upper())
        scale_hdr.setObjectName("section_label")
        scale_hdr.setAlignment(Qt.AlignmentFlag.AlignCenter)
        rl.addWidget(scale_hdr)

        scale_cols = QHBoxLayout()
        scale_cols.setSpacing(0)
        col_sx, self.lbl_scale_x = _param_col("X 方向", color=_GREEN, font_size=14)
        col_sy, self.lbl_scale_y = _param_col("Y 方向", color=_GREEN, font_size=14)
        vdiv2 = QFrame()
        vdiv2.setFrameShape(QFrame.Shape.VLine)
        vdiv2.setStyleSheet(f"color: {_BORDER};")
        scale_cols.addLayout(col_sx)
        scale_cols.addWidget(vdiv2)
        scale_cols.addLayout(col_sy)
        rl.addLayout(scale_cols)

        unit_lbl = QLabel("mm/px")
        unit_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        unit_lbl.setStyleSheet(f"color: {_TEXT_MUTED}; font-size: 10px; font-weight: 600; letter-spacing: 1px;")
        rl.addWidget(unit_lbl)

        pl.addWidget(result_card)

        root.addWidget(panel)

        self.statusBar().showMessage("就绪 — 请打开图片")

    def _build_menu(self):
        menu = self.menuBar()
        file_menu = menu.addMenu("文件")

        open_act = QAction("打开图片...", self)
        open_act.setShortcut("Ctrl+O")
        open_act.triggered.connect(self._open_image)
        file_menu.addAction(open_act)

        file_menu.addSeparator()

        quit_act = QAction("退出", self)
        quit_act.setShortcut("Ctrl+Q")
        quit_act.triggered.connect(self.close)
        file_menu.addAction(quit_act)

    # ------------------------------------------------------------------ #
    def _open_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "打开图片", "",
            "图片文件 (*.bmp *.png *.jpg *.jpeg *.tif *.tiff);;所有文件 (*)"
        )
        if not path:
            return
        if not self._is_ascii_path(path):
            QMessageBox.warning(
                self,
                "路径不支持中文",
                "当前程序暂不支持包含中文的图片路径。\n请把图片放到纯英文目录后再打开。",
            )
            self.statusBar().showMessage("打开失败：图片路径包含中文，请使用纯英文目录")
            return
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            self.statusBar().showMessage(f"无法读取文件: {path}")
            return
        self._bgr_image = img
        self.image_view.load_image(img)
        self._current_roi = None
        self.btn_rotate_left.setEnabled(True)
        self.btn_rotate_right.setEnabled(True)
        self.btn_detect.setEnabled(False)
        self.btn_clear.setEnabled(False)
        self._clear_results()
        h, w = img.shape[:2]
        self._update_image_info()
        self.statusBar().showMessage(f"已加载: {path}  ({w} × {h})")

    def _on_roi_selected(self, roi: RotatedRoi):
        self._current_roi = roi
        self.btn_detect.setEnabled(True)
        self.btn_clear.setEnabled(True)
        self.image_view.clear_detected_ellipses()
        self.statusBar().showMessage(
            f"ROI center=({roi.center_x:.1f}, {roi.center_y:.1f}) "
            f"size={roi.width:.1f} x {roi.height:.1f} px "
            f"angle={roi.angle_deg:.1f} deg"
        )

    def _detect(self):
        if self._bgr_image is None or self._current_roi is None:
            return
        crop = self._extract_rotated_roi(self._current_roi)
        if crop is None or crop.size == 0:
            self.statusBar().showMessage("ROI is invalid or outside the image")
            return
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        crop_h, crop_w = gray.shape[:2]
        if self.thr_auto.isChecked():
            thr = None
        else:
            try:
                thr = max(0, min(255, int(self.thr_input.text().strip())))
            except ValueError:
                thr = None
        ellipses = detect_ellipses(gray, threshold=thr)
        self._clear_results()
        self.image_view.clear_detected_ellipses()
        if not ellipses:
            self.statusBar().showMessage("未检测到椭圆，请调整 ROI 或检查图像")
            return

        roi = self._current_roi
        a = math.radians(roi.angle_deg)
        cos_a = math.cos(a)
        sin_a = math.sin(a)
        # getRectSubPix returns an integer-sized patch whose source-space center
        # lands at ((w - 1) / 2, (h - 1) / 2) in crop coordinates.
        crop_center_x = (crop_w - 1) / 2.0
        crop_center_y = (crop_h - 1) / 2.0
        ellipses_in_image = []
        for e in ellipses:
            cx_crop, cy_crop = e['center']
            dx = cx_crop - crop_center_x
            dy = cy_crop - crop_center_y
            orig_cx = roi.center_x + dx * cos_a - dy * sin_a
            orig_cy = roi.center_y + dx * sin_a + dy * cos_a

            contour_image = []
            for pt in e['contour']:
                pdx = float(pt[0]) - crop_center_x
                pdy = float(pt[1]) - crop_center_y
                contour_image.append((
                    roi.center_x + pdx * cos_a - pdy * sin_a,
                    roi.center_y + pdx * sin_a + pdy * cos_a,
                ))

            ellipses_in_image.append({
                'center_x': orig_cx,
                'center_y': orig_cy,
                'x_axis': e['x_axis'],
                'y_axis': e['y_axis'],
                'angle_deg': e['angle'] + roi.angle_deg,
                'contour_points': contour_image,
            })
        self.image_view.set_detected_ellipses(ellipses_in_image)

        self.table.setRowCount(len(ellipses))
        ratios = []
        for i, e in enumerate(ellipses):
            self.table.setItem(i, 0, self._cell(str(i + 1)))
            self.table.setItem(i, 1, self._cell(f"{e['x_axis']:.2f}"))
            self.table.setItem(i, 2, self._cell(f"{e['y_axis']:.2f}"))
            self.table.setItem(i, 3, self._cell(f"{e['ratio']:.4f}"))
            ratios.append(e["ratio"])
        avg = float(np.mean(ratios))
        post_div, mult = best_fraction(avg)   # 后分配器(1-128), 乘法器(1-32)
        self.avg_label.setText(f"{avg:.4f}")
        self.lbl_mult.setText(str(mult))
        self.lbl_post_div.setText(str(post_div))
        msg = f"检测到 {len(ellipses)} 个椭圆    平均 Y/X: {avg:.4f}    乘法器 {mult}  后分配器 {post_div}"
        self.statusBar().showMessage(msg)

        avg_x = float(np.mean([e['x_axis'] for e in ellipses]))
        avg_y = float(np.mean([e['y_axis'] for e in ellipses]))
        self._update_scale(avg_x, avg_y)

    def _update_scale(self, avg_x_px: float, avg_y_px: float):
        text = self.circle_diam_input.text().strip()
        if not text:
            self.lbl_scale_x.setText("—")
            self.lbl_scale_y.setText("—")
            return
        try:
            real_mm = float(text)
            if real_mm <= 0:
                raise ValueError
        except ValueError:
            self.lbl_scale_x.setText("—")
            self.lbl_scale_y.setText("—")
            return
        sx = real_mm / avg_x_px
        sy = real_mm / avg_y_px
        self.lbl_scale_x.setText(f"{sx:.4f}")
        self.lbl_scale_y.setText(f"{sy:.4f}")

    def _on_scale_input_changed(self, _text: str):
        # Re-read table to recalculate scale without re-detecting
        rows = self.table.rowCount()
        if rows == 0:
            self.lbl_scale_x.setText("—")
            self.lbl_scale_y.setText("—")
            return
        x_vals, y_vals = [], []
        for i in range(rows):
            try:
                x_vals.append(float(self.table.item(i, 1).text()))
                y_vals.append(float(self.table.item(i, 2).text()))
            except (AttributeError, ValueError):
                pass
        if x_vals and y_vals:
            self._update_scale(float(np.mean(x_vals)), float(np.mean(y_vals)))

    def _on_exposure_changed(self, text: str):
        text = text.strip()
        if not text:
            self.freq_label.setText("—")
            return
        try:
            exposure_us = float(text)
            if exposure_us <= 0:
                raise ValueError
        except ValueError:
            self.freq_label.setText("—")
            return
        freq_hz = 1_000_000.0 / exposure_us
        self.freq_label.setText(f"{freq_hz:.1f} Hz")

    def _rotate_left(self):
        self._rotate_image(clockwise=False)

    def _rotate_right(self):
        self._rotate_image(clockwise=True)

    def _rotate_image(self, clockwise: bool):
        if self._bgr_image is None:
            return
        direction = "右旋转" if clockwise else "左旋转"
        rotate_flag = (
            cv2.ROTATE_90_CLOCKWISE if clockwise else cv2.ROTATE_90_COUNTERCLOCKWISE
        )
        self._bgr_image = cv2.rotate(self._bgr_image, rotate_flag)
        self.image_view.load_image(self._bgr_image)
        self._current_roi = None
        self.btn_detect.setEnabled(False)
        self.btn_clear.setEnabled(False)
        self._clear_results()
        self._update_image_info()
        h, w = self._bgr_image.shape[:2]
        self.statusBar().showMessage(f"图片已{direction}，当前尺寸 {w} x {h}")

    def _clear_roi(self):
        self.image_view.clear_roi()
        self._current_roi = None
        self.btn_detect.setEnabled(False)
        self.btn_clear.setEnabled(False)
        self._clear_results()
        self.statusBar().showMessage("ROI 已清除")

    # ------------------------------------------------------------------ #
    def _clear_results(self):
        self.table.setRowCount(0)
        self.avg_label.setText("—")
        self.lbl_mult.setText("—")
        self.lbl_post_div.setText("—")
        self.lbl_scale_x.setText("—")
        self.lbl_scale_y.setText("—")

    def _update_image_info(self):
        if self._bgr_image is None:
            self.lbl_img_x.setText("—")
            self.lbl_img_y.setText("—")
            return

        h, w = self._bgr_image.shape[:2]
        self.lbl_img_x.setText(f"{w} px")
        self.lbl_img_y.setText(f"{h} px")

    @staticmethod
    def _is_ascii_path(path: str) -> bool:
        return path.isascii()

    def _extract_rotated_roi(self, roi: RotatedRoi) -> np.ndarray | None:
        if self._bgr_image is None:
            return None

        width = max(1, int(round(roi.width)))
        height = max(1, int(round(roi.height)))
        center = (float(roi.center_x), float(roi.center_y))

        image_h, image_w = self._bgr_image.shape[:2]
        if not (0.0 <= center[0] <= image_w and 0.0 <= center[1] <= image_h):
            return None

        rotate_mat = cv2.getRotationMatrix2D(center, roi.angle_deg, 1.0)
        rotated = cv2.warpAffine(
            self._bgr_image,
            rotate_mat,
            (image_w, image_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )
        return cv2.getRectSubPix(rotated, (width, height), center)

    @staticmethod
    def _cell(text: str) -> QTableWidgetItem:
        item = QTableWidgetItem(text)
        item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        return item
