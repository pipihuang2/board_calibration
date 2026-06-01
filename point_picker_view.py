from __future__ import annotations

import math

import numpy as np
from PyQt6.QtCore import QPoint, QPointF, QRect, QRectF, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QImage, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import QWidget


class PointPickerView(QWidget):
    """Image widget for placing, dragging, and displaying numbered calibration points."""

    points_changed = pyqtSignal(list)             # list[dict{"id": int, "x": float, "y": float}]
    mouse_pos_changed = pyqtSignal(float, float)  # image-space (x, y); (-1,-1) = off image

    _POINT_RADIUS = 7.0
    _HIT_RADIUS   = 12.0

    def __init__(self, parent=None):
        super().__init__(parent)
        self._pixmap: QPixmap | None = None
        self._img_w = 0
        self._img_h = 0
        self._zoom  = 1.0
        self._view_cx = 0.0
        self._view_cy = 0.0

        self._points: list[dict] = []   # [{"id": int, "x": float, "y": float}]
        self._next_id = 1
        self._dragging_idx: int | None = None
        self._drag_offset = QPointF()
        self._real_coords_by_id: dict[int, tuple[float, float]] = {}

        self._panning = False
        self._pan_start = QPointF()
        self._pan_start_cx = 0.0
        self._pan_start_cy = 0.0

        self.setMinimumSize(400, 300)
        self.setMouseTracking(True)
        self.setCursor(Qt.CursorShape.CrossCursor)

    # ── Public API ──────────────────────────────────────────────────── #
    def load_image(self, bgr_array: np.ndarray):
        h, w = bgr_array.shape[:2]
        rgb = bgr_array[..., ::-1].copy()
        qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888)
        self._pixmap = QPixmap.fromImage(qimg)
        self._img_w, self._img_h = w, h
        self._zoom = 1.0
        self._view_cx = w / 2.0
        self._view_cy = h / 2.0
        self._points.clear()
        self._next_id = 1
        self.update()
        self.points_changed.emit([])

    def get_points(self) -> list[dict]:
        return [dict(p) for p in self._points]

    def set_points_silent(self, points: list[dict]):
        """Update internal point list without emitting points_changed (avoids table↔view loop)."""
        self._points = [dict(p) for p in points]
        self._next_id = (max(p["id"] for p in self._points) + 1) if self._points else 1
        self.update()

    def clear_points(self):
        self._points.clear()
        self._real_coords_by_id.clear()
        self._next_id = 1
        self.update()
        self.points_changed.emit([])

    def set_real_coords(self, coords_by_id: dict[int, tuple[float, float]]):
        self._real_coords_by_id = dict(coords_by_id)
        self.update()

    # ── Mouse events ────────────────────────────────────────────────── #
    def mousePressEvent(self, event):
        if self._pixmap is None:
            return

        if event.button() == Qt.MouseButton.RightButton:
            self._panning = True
            self._pan_start = event.position()
            self._pan_start_cx = self._view_cx
            self._pan_start_cy = self._view_cy
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            return

        if event.button() != Qt.MouseButton.LeftButton:
            return
        pos_w = event.position()
        hit = self._point_at(pos_w)
        if hit is not None:
            self._dragging_idx = hit
            p = self._points[hit]
            cw = self._img_to_widget(QPointF(p["x"], p["y"]))
            self._drag_offset = QPointF(cw.x() - pos_w.x(), cw.y() - pos_w.y())
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
        elif self._in_image(event.pos()):
            img_pt = self._widget_to_img(pos_w)
            self._points.append({"id": self._next_id, "x": img_pt.x(), "y": img_pt.y()})
            self._next_id += 1
            self.update()
            self.points_changed.emit(self.get_points())

    def mouseMoveEvent(self, event):
        pos_w = event.position()

        if self._panning:
            scale = self._fit_scale() * self._zoom
            if scale > 0:
                dx = (pos_w.x() - self._pan_start.x()) / scale
                dy = (pos_w.y() - self._pan_start.y()) / scale
                self._view_cx = self._pan_start_cx - dx
                self._view_cy = self._pan_start_cy - dy
                self._clamp_view()
                self.update()
            if self._in_image(event.pos()):
                ip = self._widget_to_img(pos_w)
                self.mouse_pos_changed.emit(ip.x(), ip.y())
            else:
                self.mouse_pos_changed.emit(-1.0, -1.0)
            return

        if self._in_image(event.pos()):
            ip = self._widget_to_img(pos_w)
            self.mouse_pos_changed.emit(ip.x(), ip.y())
        else:
            self.mouse_pos_changed.emit(-1.0, -1.0)

        if self._dragging_idx is not None:
            adj = QPointF(pos_w.x() + self._drag_offset.x(),
                          pos_w.y() + self._drag_offset.y())
            ip = self._widget_to_img(adj)
            p = self._points[self._dragging_idx]
            p["x"] = max(0.0, min(float(self._img_w), ip.x()))
            p["y"] = max(0.0, min(float(self._img_h), ip.y()))
            self.update()
            self.points_changed.emit(self.get_points())
        else:
            cursor = (Qt.CursorShape.OpenHandCursor if self._point_at(pos_w) is not None
                      else Qt.CursorShape.CrossCursor)
            self.setCursor(cursor)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.RightButton:
            self._panning = False
            cursor = (Qt.CursorShape.OpenHandCursor if self._point_at(event.position()) is not None
                      else Qt.CursorShape.CrossCursor)
            self.setCursor(cursor)
            return
        if event.button() != Qt.MouseButton.LeftButton:
            return
        self._dragging_idx = None
        cursor = (Qt.CursorShape.OpenHandCursor if self._point_at(event.position()) is not None
                  else Qt.CursorShape.CrossCursor)
        self.setCursor(cursor)

    def leaveEvent(self, event):
        self.mouse_pos_changed.emit(-1.0, -1.0)
        super().leaveEvent(event)

    def wheelEvent(self, event):
        if self._pixmap is None:
            return
        delta = event.angleDelta().y()
        if delta == 0:
            return
        step = 1.15 if delta > 0 else 1 / 1.15
        new_zoom = min(20.0, max(1.0, self._zoom * step))
        if abs(new_zoom - self._zoom) < 1e-6:
            return
        old_scale, ox, oy = self._transform()
        cur = event.position()
        img_x = (cur.x() - ox) / old_scale
        img_y = (cur.y() - oy) / old_scale
        self._zoom = new_zoom
        ns = self._fit_scale() * self._zoom
        self._view_cx = img_x - (cur.x() - self.width() / 2) / ns
        self._view_cy = img_y - (cur.y() - self.height() / 2) / ns
        self._clamp_view()
        self.update()
        event.accept()

    # ── Painting ────────────────────────────────────────────────────── #
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        if self._pixmap is None:
            painter.fillRect(self.rect(), QColor(234, 236, 240))
            painter.setPen(QPen(QColor(180, 185, 195), 2, Qt.PenStyle.DashLine))
            margin = 24
            painter.drawRoundedRect(
                self.rect().adjusted(margin, margin, -margin, -margin), 12, 12)
            painter.setPen(QColor(150, 155, 165))
            f = painter.font(); f.setPointSize(13); painter.setFont(f)
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter,
                             "打开图片后点击放置标定点")
            return

        scale, ox, oy = self._transform()
        painter.drawPixmap(int(ox), int(oy),
                           int(self._img_w * scale), int(self._img_h * scale),
                           self._pixmap)

        id_font = QFont(); id_font.setPointSize(9); id_font.setBold(True)
        real_font = QFont(); real_font.setPointSize(9)

        for i, p in enumerate(self._points):
            center = self._img_to_widget(QPointF(p["x"], p["y"]))
            r = self._POINT_RADIUS
            dragging = (i == self._dragging_idx)
            fill    = QColor(255, 120,  0, 200) if dragging else QColor(255, 200,  0, 180)
            outline = QColor(255,  80,  0)       if dragging else QColor(200, 140,  0)

            painter.save()

            # ── Circle ───────────────────────────────────────────────
            painter.setPen(QPen(outline, 2.0))
            painter.setBrush(fill)
            painter.drawEllipse(center, r, r)

            # ── ID label (centered inside circle) ────────────────────
            painter.setFont(id_font)
            fm_id = painter.fontMetrics()
            label = str(p["id"])
            lx = center.x() - fm_id.horizontalAdvance(label) / 2
            ly = center.y() + fm_id.ascent() / 2 - 1
            painter.setPen(QColor(0, 0, 0))
            for ddx, ddy in ((-1, -1), (1, -1), (-1, 1), (1, 1)):
                painter.drawText(QPointF(lx + ddx, ly + ddy), label)
            painter.setPen(QColor(255, 255, 255))
            painter.drawText(QPointF(lx, ly), label)

            # ── Real coord label (floating tag to upper-right) ────────
            real = self._real_coords_by_id.get(p["id"])
            if real is not None:
                painter.setFont(real_font)
                fm_r = painter.fontMetrics()
                tag = f"({real[0]:.2f}, {real[1]:.2f})"
                tw = fm_r.horizontalAdvance(tag)
                th = fm_r.height()
                pad_x, pad_y = 5, 3
                # tag_x/tag_y = top-left corner of the text content area
                tag_x = center.x() + r + 6
                tag_y = center.y() - r
                # background: pad around the content area
                bg = QRectF(tag_x - pad_x, tag_y - pad_y,
                            tw + pad_x * 2, th + pad_y * 2)
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(QColor(0, 0, 0, 170))
                painter.drawRoundedRect(bg, 3, 3)
                # text baseline = top of content area + ascent
                painter.setPen(QColor(16, 185, 129))
                painter.drawText(QPointF(tag_x, tag_y + fm_r.ascent()), tag)

            painter.restore()

    def resizeEvent(self, event):
        self.update()

    # ── Geometry helpers ─────────────────────────────────────────────── #
    def _point_at(self, pos_w: QPointF) -> int | None:
        for i, p in enumerate(self._points):
            c = self._img_to_widget(QPointF(p["x"], p["y"]))
            if math.hypot(pos_w.x() - c.x(), pos_w.y() - c.y()) <= self._HIT_RADIUS:
                return i
        return None

    def _in_image(self, pos: QPoint) -> bool:
        if self._pixmap is None:
            return False
        scale, ox, oy = self._transform()
        rect = QRect(int(round(ox)), int(round(oy)),
                     int(round(self._img_w * scale)), int(round(self._img_h * scale)))
        return rect.contains(pos)

    def _widget_to_img(self, pos: QPointF) -> QPointF:
        scale, ox, oy = self._transform()
        return QPointF(
            max(0.0, min(float(self._img_w), (pos.x() - ox) / scale)),
            max(0.0, min(float(self._img_h), (pos.y() - oy) / scale)),
        )

    def _img_to_widget(self, pos: QPointF) -> QPointF:
        scale, ox, oy = self._transform()
        return QPointF(ox + pos.x() * scale, oy + pos.y() * scale)

    def _transform(self):
        if self._pixmap is None or self._img_w == 0:
            return 1.0, 0.0, 0.0
        scale = self._fit_scale() * self._zoom
        self._clamp_view(scale)
        ww, wh = self.width(), self.height()
        ox = ww / 2 - self._view_cx * scale
        oy = wh / 2 - self._view_cy * scale
        dw, dh = self._img_w * scale, self._img_h * scale
        ox = (ww - dw) / 2 if dw <= ww else min(0.0, max(ww - dw, ox))
        oy = (wh - dh) / 2 if dh <= wh else min(0.0, max(wh - dh, oy))
        return scale, ox, oy

    def _fit_scale(self) -> float:
        if self._pixmap is None or self._img_w == 0 or self._img_h == 0:
            return 1.0
        return min(max(1, self.width()) / self._img_w,
                   max(1, self.height()) / self._img_h)

    def _clamp_view(self, scale: float | None = None):
        if self._pixmap is None:
            return
        scale = scale or (self._fit_scale() * self._zoom)
        ww, wh = self.width(), self.height()
        hw, hh = ww / (2 * scale), wh / (2 * scale)
        self._view_cx = (self._img_w / 2 if self._img_w * scale <= ww
                         else min(self._img_w - hw, max(hw, self._view_cx)))
        self._view_cy = (self._img_h / 2 if self._img_h * scale <= wh
                         else min(self._img_h - hh, max(hh, self._view_cy)))
