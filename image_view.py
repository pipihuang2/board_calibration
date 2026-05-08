from __future__ import annotations
import math
import numpy as np
from PyQt6.QtCore import Qt, QRect, QPoint, QPointF, pyqtSignal
from PyQt6.QtGui import QPixmap, QPainter, QColor, QPen, QImage, QPolygonF, QFont, QBrush
from PyQt6.QtWidgets import QWidget


class ImageView(QWidget):
    roi_selected = pyqtSignal(QRect)  # emits ROI in image coordinates

    def __init__(self, parent=None):
        super().__init__(parent)
        self._pixmap: QPixmap | None = None
        self._img_w = 0
        self._img_h = 0

        self._drawing = False
        self._start_widget = QPoint()
        self._end_widget = QPoint()
        self._roi_widget: QRect | None = None

        self.setMinimumSize(400, 300)
        self.setCursor(Qt.CursorShape.CrossCursor)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def load_image(self, bgr_array: np.ndarray):
        h, w = bgr_array.shape[:2]
        rgb = bgr_array[..., ::-1].copy()
        qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888)
        self._pixmap = QPixmap.fromImage(qimg)
        self._img_w, self._img_h = w, h
        self._roi_widget = None
        self.update()

    def clear_roi(self):
        self._roi_widget = None
        self.update()

    def roi_in_image_coords(self) -> QRect | None:
        if self._roi_widget is None:
            return None
        scale, ox, oy = self._transform()
        r = self._roi_widget.normalized()
        ix = int((r.x() - ox) / scale)
        iy = int((r.y() - oy) / scale)
        iw = int(r.width() / scale)
        ih = int(r.height() / scale)
        return QRect(ix, iy, iw, ih).normalized()

    # ------------------------------------------------------------------ #
    # Mouse events — ROI drawing
    # ------------------------------------------------------------------ #
    def mousePressEvent(self, event):
        if self._pixmap is None:
            return
        if event.button() == Qt.MouseButton.LeftButton:
            self._drawing = True
            self._start_widget = event.pos()
            self._end_widget = event.pos()

    def mouseMoveEvent(self, event):
        if self._drawing:
            self._end_widget = event.pos()
            self.update()

    def mouseReleaseEvent(self, event):
        if self._drawing and event.button() == Qt.MouseButton.LeftButton:
            self._drawing = False
            self._end_widget = event.pos()
            self._roi_widget = QRect(self._start_widget, self._end_widget).normalized()
            self.update()
            roi_img = self.roi_in_image_coords()
            if roi_img and roi_img.width() > 5 and roi_img.height() > 5:
                self.roi_selected.emit(roi_img)

    # ------------------------------------------------------------------ #
    # Painting
    # ------------------------------------------------------------------ #
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)

        if self._pixmap is None:
            painter.fillRect(self.rect(), QColor(234, 236, 240))
            # dashed border
            pen = QPen(QColor(180, 185, 195), 2, Qt.PenStyle.DashLine)
            painter.setPen(pen)
            margin = 24
            painter.drawRoundedRect(
                self.rect().adjusted(margin, margin, -margin, -margin), 12, 12
            )
            painter.setPen(QColor(150, 155, 165))
            font = painter.font()
            font.setPointSize(13)
            painter.setFont(font)
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter,
                             "打开图片后\n在此拖拽选取 ROI")
            return

        scale, ox, oy = self._transform()
        draw_w = int(self._img_w * scale)
        draw_h = int(self._img_h * scale)
        painter.drawPixmap(int(ox), int(oy), draw_w, draw_h, self._pixmap)

        # ROI overlay
        roi_rect = None
        if self._drawing:
            roi_rect = QRect(self._start_widget, self._end_widget).normalized()
        elif self._roi_widget:
            roi_rect = self._roi_widget

        if roi_rect:
            fill = QColor(0, 120, 255, 40)
            painter.fillRect(roi_rect, fill)
            pen = QPen(QColor(0, 120, 255), 2, Qt.PenStyle.SolidLine)
            painter.setPen(pen)
            painter.drawRect(roi_rect)

        self._draw_axes(painter, int(ox), int(oy), draw_w, draw_h)

    def resizeEvent(self, event):
        self.update()

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _draw_axes(self, painter: QPainter, ox: int, oy: int, dw: int, dh: int):
        """Draw X/Y direction arrows in the top-left corner of the image."""
        ARROW_LEN = 44
        HEAD_LEN  = 8
        HEAD_W    = 4
        LINE_W    = 2.0
        PAD       = 12   # inner padding inside the box

        # Fixed box size — large enough to contain arrows + Chinese labels
        BOX_W, BOX_H = 148, 88
        box_x = ox + 10
        box_y = oy + 10

        # Origin of the L-shaped axes sits at bottom-left inside the box
        origin_x = box_x + PAD + 4
        origin_y = box_y + BOX_H - PAD

        # Background box
        painter.save()
        painter.setBrush(QBrush(QColor(0, 0, 0, 110)))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRoundedRect(QRect(box_x, box_y, BOX_W, BOX_H), 8, 8)
        painter.restore()

        font = QFont()
        font.setPointSize(8)
        font.setBold(True)

        def draw_arrow(x0, y0, x1, y1, color, label, label_dx, label_dy):
            painter.save()
            pen = QPen(QColor(*color), LINE_W)
            pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            painter.setPen(pen)
            painter.drawLine(QPointF(x0, y0), QPointF(x1, y1))

            dx, dy = x1 - x0, y1 - y0
            length = math.hypot(dx, dy)
            ux, uy = dx / length, dy / length
            px, py = -uy, ux

            tip  = QPointF(x1, y1)
            base = QPointF(x1 - ux * HEAD_LEN, y1 - uy * HEAD_LEN)
            p1   = QPointF(base.x() + px * HEAD_W, base.y() + py * HEAD_W)
            p2   = QPointF(base.x() - px * HEAD_W, base.y() - py * HEAD_W)
            painter.setBrush(QBrush(QColor(*color)))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawPolygon(QPolygonF([tip, p1, p2]))

            painter.setFont(font)
            painter.setPen(QColor(*color))
            painter.drawText(QPointF(x1 + label_dx, y1 + label_dy), label)
            painter.restore()

        # X → right  (运动方向), label sits below the tip
        draw_arrow(origin_x, origin_y,
                   origin_x + ARROW_LEN, origin_y,
                   (80, 200, 120), "X 运动方向", -ARROW_LEN // 2, 14)

        # Y ↑ up  (拍摄方向), label sits to the right of the tip
        draw_arrow(origin_x, origin_y,
                   origin_x, origin_y - ARROW_LEN,
                   (100, 160, 255), "Y 拍摄方向", 6, 4)

    def _transform(self):
        """Return (scale, offset_x, offset_y) for centered fit."""
        if self._pixmap is None or self._img_w == 0:
            return 1.0, 0.0, 0.0
        ww, wh = self.width(), self.height()
        scale = min(ww / self._img_w, wh / self._img_h)
        ox = (ww - self._img_w * scale) / 2
        oy = (wh - self._img_h * scale) / 2
        return scale, ox, oy
