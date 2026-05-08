from __future__ import annotations
import cv2
import numpy as np
from fractions import Fraction


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
from PyQt6.QtCore import Qt, QRect
from PyQt6.QtGui import QAction, QFont, QColor, QPalette
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QPushButton, QLabel, QTableWidget, QTableWidgetItem,
    QFileDialog, QHeaderView, QFrame, QGraphicsDropShadowEffect,
    QLineEdit, QSizePolicy,
)

from image_view import ImageView
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
        self._current_roi: QRect | None = None

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

        # Buttons
        self.btn_open = QPushButton("打开图片")
        self.btn_open.clicked.connect(self._open_image)

        self.btn_detect = QPushButton("检测椭圆")
        self.btn_detect.setEnabled(False)
        self.btn_detect.clicked.connect(self._detect)

        self.btn_clear = QPushButton("清除 ROI")
        self.btn_clear.setObjectName("btn_clear")
        self.btn_clear.setEnabled(False)
        self.btn_clear.clicked.connect(self._clear_roi)

        for btn in (self.btn_open, self.btn_detect, self.btn_clear):
            pl.addWidget(btn)

        pl.addSpacing(4)

        # ── Image info card ───────────────────────────────────────────
        img_card = QFrame()
        img_card.setObjectName("metric_card")
        ic_lay = QVBoxLayout(img_card)
        ic_lay.setContentsMargins(14, 10, 14, 10)
        ic_lay.setSpacing(6)

        ic_lay.addWidget(_section_label("图像信息"))

        def _info_row(label_text):
            row = QHBoxLayout()
            row.setSpacing(0)
            key = QLabel(label_text)
            key.setStyleSheet(f"color: {_TEXT_MUTED}; font-size: 12px;")
            val = QLabel("—")
            val.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            val.setStyleSheet(f"color: {_TEXT}; font-size: 12px; font-weight: 600;")
            row.addWidget(key)
            row.addStretch()
            row.addWidget(val)
            return row, val

        row_x, self.lbl_img_x = _info_row("X 方向（运动）")
        row_y, self.lbl_img_y = _info_row("Y 方向（拍摄）")
        ic_lay.addLayout(row_x)
        ic_lay.addLayout(row_y)
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
        self.table.setMinimumHeight(160)
        pl.addWidget(self.table, stretch=1)

        pl.addSpacing(4)

        # ── Avg ratio card ────────────────────────────────────────────
        avg_card = QFrame()
        avg_card.setObjectName("metric_card")
        avg_lay = QVBoxLayout(avg_card)
        avg_lay.setContentsMargins(14, 10, 14, 12)
        avg_lay.setSpacing(6)

        avg_hdr = QLabel("平均 Y / X 比值".upper())
        avg_hdr.setObjectName("section_label")
        avg_hdr.setAlignment(Qt.AlignmentFlag.AlignCenter)
        avg_lay.addWidget(avg_hdr)

        self.avg_label = QLabel("—")
        f1 = QFont(); f1.setPointSize(20); f1.setBold(True)
        self.avg_label.setFont(f1)
        self.avg_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.avg_label.setStyleSheet(f"color: {_ACCENT};")
        avg_lay.addWidget(self.avg_label)

        # Divider inside card
        inner_sep = QFrame()
        inner_sep.setFrameShape(QFrame.Shape.HLine)
        inner_sep.setStyleSheet(f"color: {_BORDER};")
        avg_lay.addWidget(inner_sep)

        # Two-column: 乘法器 | 后分配器
        cols = QHBoxLayout()
        cols.setSpacing(0)

        def _param_col(title: str) -> tuple[QVBoxLayout, QLabel]:
            col = QVBoxLayout()
            col.setSpacing(2)
            val_lbl = QLabel("—")
            fv = QFont(); fv.setPointSize(20); fv.setBold(True)
            val_lbl.setFont(fv)
            val_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            val_lbl.setStyleSheet(f"color: {_TEXT};")
            ttl_lbl = QLabel(title)
            ttl_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            ttl_lbl.setObjectName("section_label")
            col.addWidget(val_lbl)
            col.addWidget(ttl_lbl)
            return col, val_lbl

        col_mult, self.lbl_mult     = _param_col("乘法器")
        col_post, self.lbl_post_div = _param_col("后分配器")

        # Vertical divider between columns
        vdiv = QFrame()
        vdiv.setFrameShape(QFrame.Shape.VLine)
        vdiv.setStyleSheet(f"color: {_BORDER};")

        cols.addLayout(col_mult)
        cols.addWidget(vdiv)
        cols.addLayout(col_post)
        avg_lay.addLayout(cols)

        pl.addWidget(avg_card)

        pl.addSpacing(2)

        # ── Camera params card ────────────────────────────────────────
        cam_card = QFrame()
        cam_card.setObjectName("metric_card")
        cl = QVBoxLayout(cam_card)
        cl.setContentsMargins(14, 10, 14, 12)
        cl.setSpacing(6)

        cam_hdr = QLabel("相机参数".upper())
        cam_hdr.setObjectName("section_label")
        cl.addWidget(cam_hdr)

        # Exposure row
        exp_row = QHBoxLayout()
        exp_row.setSpacing(8)
        exp_lbl = QLabel("曝光时间")
        exp_lbl.setStyleSheet(f"color: {_TEXT_MUTED}; font-size: 12px;")
        exp_row.addWidget(exp_lbl)
        self.exp_input = QLineEdit()
        self.exp_input.setPlaceholderText("μs")
        self.exp_input.textChanged.connect(self._on_exposure_changed)
        exp_row.addWidget(self.exp_input)
        cl.addLayout(exp_row)

        freq_hdr = QLabel("最大采集频率".upper())
        freq_hdr.setObjectName("section_label")
        freq_hdr.setAlignment(Qt.AlignmentFlag.AlignCenter)
        cl.addWidget(freq_hdr)

        self.freq_label = QLabel("—")
        f2 = QFont()
        f2.setPointSize(20)
        f2.setBold(True)
        self.freq_label.setFont(f2)
        self.freq_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.freq_label.setStyleSheet(f"color: {_GREEN};")
        cl.addWidget(self.freq_label)

        pl.addWidget(cam_card)

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
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            self.statusBar().showMessage(f"无法读取文件: {path}")
            return
        self._bgr_image = img
        self.image_view.load_image(img)
        self._current_roi = None
        self.btn_detect.setEnabled(False)
        self.btn_clear.setEnabled(False)
        self._clear_results()
        h, w = img.shape[:2]
        self.lbl_img_x.setText(f"{w} px")
        self.lbl_img_y.setText(f"{h} px")
        self.statusBar().showMessage(f"已加载: {path}  ({w} × {h})")

    def _on_roi_selected(self, roi: QRect):
        self._current_roi = roi
        self.btn_detect.setEnabled(True)
        self.btn_clear.setEnabled(True)
        self.statusBar().showMessage(
            f"ROI: ({roi.x()}, {roi.y()})  {roi.width()} × {roi.height()} px  —  点击「检测椭圆」"
        )

    def _detect(self):
        if self._bgr_image is None or self._current_roi is None:
            return
        roi = self._current_roi
        x, y, w, h = roi.x(), roi.y(), roi.width(), roi.height()
        ih, iw = self._bgr_image.shape[:2]
        x = max(0, x); y = max(0, y)
        w = min(w, iw - x); h = min(h, ih - y)
        if w <= 0 or h <= 0:
            self.statusBar().showMessage("ROI 超出图像范围")
            return
        crop = self._bgr_image[y:y+h, x:x+w]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        ellipses = detect_ellipses(gray)
        self._clear_results()
        if not ellipses:
            self.statusBar().showMessage("未检测到椭圆，请调整 ROI 或检查图像")
            return
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
        self.statusBar().showMessage(
            f"检测到 {len(ellipses)} 个椭圆    平均 Y/X: {avg:.4f}    乘法器 {mult}  后分配器 {post_div}"
        )

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

    @staticmethod
    def _cell(text: str) -> QTableWidgetItem:
        item = QTableWidgetItem(text)
        item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        return item
