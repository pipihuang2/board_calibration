import cv2
import numpy as np


def detect_ellipses(gray_img: np.ndarray, min_area: int = 200, max_area: int = 500000) -> list[dict]:
    blurred = cv2.GaussianBlur(gray_img, (5, 5), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    results = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area:
            continue
        if len(cnt) < 5:
            continue

        (cx, cy), (minor, major), angle = cv2.fitEllipse(cnt)

        if minor < 1:
            continue

        # cv2.fitEllipse angle = rotation of major axis from vertical (Y-axis), clockwise.
        # When angle < 45 or > 135, major axis is closer to vertical (Y/拍摄方向).
        if angle < 45 or angle > 135:
            y_axis = major
            x_axis = minor
        else:
            y_axis = minor
            x_axis = major

        if x_axis < 1:
            continue

        ratio = y_axis / x_axis   # Y(拍摄方向) / X(运动方向)
        if ratio > 10 or ratio < 0.1:
            continue

        results.append({
            "center": (cx, cy),
            "x_axis": x_axis,
            "y_axis": y_axis,
            "ratio": ratio,
            "angle": angle,
        })

    return results
