import json
import math

import numpy as np

_Y_CENTER = 4096.0  # 线扫传感器 y 轴中心，拟合与变换必须保持一致


def fit_custom_transform_pix_to_world(points_A, points_B):
    """

    """
    points_A = np.asarray(points_A, dtype=np.float64)
    points_B = np.asarray(points_B, dtype=np.float64)
    if points_A.shape != points_B.shape or points_A.shape[1] != 2:
        print(f'len(points_A) != len(points_B), {points_A.shape}, {points_B.shape}')
        raise ValueError("points_A 和 points_B 必须是形状相同的 (N, 2) 数组")

    xA, yA = points_A[:, 0], points_A[:, 1]
    xB, yB = points_B[:, 0], points_B[:, 1]

    # 对于 x' = a * x + b * y + f
    design_x = np.c_[xA, yA, np.ones_like(xA)]

    # 对于 y' = g * x + h * y + i * (y - Y_CENTER)^2 + j * (y - Y_CENTER)^3 + m
    design_y = np.c_[xA,
                     yA,
                     (yA - _Y_CENTER) ** 2,
                     (yA - _Y_CENTER) ** 3,
                     np.ones_like(xA)]

    y_params, _, _, _ = np.linalg.lstsq(design_y, yB, rcond=None)
    x_params, _, _, _ = np.linalg.lstsq(design_x, xB, rcond=None)

    print(f"x_params: {x_params}, y_params: {y_params}")
    return x_params, y_params


def transform_point(point, x_params, y_params):
    """
    用拟合参数将A组的点转换为B组
    point: (x, y)
    x_params: [a, b, f]
    y_params: [g, h, i, j, m]
    """
    x, y = point
    a, b, f = x_params
    x_new = a * x + b * y + f
    g, h, i, j, m = y_params
    y_new = g * x + h * y + i * (y - _Y_CENTER) ** 2 + j * (y - _Y_CENTER) ** 3 + m
    return np.array([x_new, y_new])


def pixel_to_world(point_pixel, x_params_p2w, y_params_p2w):
    """
    像素坐标 -> 真实坐标
    point_pixel: (x, y) 像素坐标
    x_params_p2w, y_params_p2w: 像素->真实 拟合出来的参数
    """
    return transform_point(point_pixel, x_params_p2w, y_params_p2w)

def calc_mse(points_A, points_B, x_params, y_params, plot: bool = True):
    """
    计算拟合欧式距离误差，打印排序表，可选绘图。
    返回：(mean_dist, per_point_errors)
      mean_dist        -- 平均欧式距离误差（标量）
      per_point_errors -- list of (original_idx, pred_pt, true_pt, distance)，按误差从大到小
    """
    points_A = np.asarray(points_A)
    points_B = np.asarray(points_B)
    pred = np.array([transform_point(p, x_params, y_params) for p in points_A])
    mse_list = []
    for idx, (pred_pt, true_pt) in enumerate(zip(pred, points_B)):
        distance = math.hypot(pred_pt[0] - true_pt[0], pred_pt[1] - true_pt[1])
        mse_list.append((idx, pred_pt, true_pt, distance))
    mse_list_sorted = sorted(mse_list, key=lambda x: x[3], reverse=True)

    print(f"{'序号':<4} {'原始坐标':<20} {'预测点':<20} {'真实点':<20} {'欧式距离':<12} {'分量误差':<20}")
    print("-" * 100)
    for idx, pred_pt, true_pt, dist in mse_list_sorted:
        original_point = points_A[idx]
        component_err = pred_pt - true_pt
        orig_str = f"[{original_point[0]:.2f}, {original_point[1]:.2f}]"
        pred_str = f"[{pred_pt[0]:.2f}, {pred_pt[1]:.2f}]"
        true_str = f"[{true_pt[0]:.2f}, {true_pt[1]:.2f}]"
        err_str  = f"[{component_err[0]:.2f}, {component_err[1]:.2f}]"
        print(f"{idx:<4} {orig_str:<20} {pred_str:<20} {true_str:<20} {dist:<12.4f} {err_str:<20}")

    distances = np.array([row[3] for row in mse_list])
    mean_dist = float(np.mean(distances))

    if plot:
        import matplotlib.pyplot as plt
        plt.rcParams['font.family'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        plt.figure(figsize=(8, 8))
        plt.scatter(points_B[:, 0], points_B[:, 1], c='b', label='真实点', marker='o')
        plt.scatter(pred[:, 0], pred[:, 1], c='r', label='预测点', marker='x')
        for p_pred, p_true in zip(pred, points_B):
            plt.plot([p_pred[0], p_true[0]], [p_pred[1], p_true[1]], 'g--', linewidth=0.5)
        plt.legend()
        plt.title("真实点与预测点对比")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.gca().set_aspect('equal', adjustable='box')
        plt.show()

    return mean_dist, mse_list_sorted


def save_params(x_params, y_params, filename):
    params = {'x_params': x_params.tolist(), 'y_params': y_params.tolist()}
    with open(filename, 'w') as f:
        json.dump(params, f)


def load_params(filename):
    with open(filename, 'r') as f:
        params = json.load(f)
    return np.array(params['x_params']), np.array(params['y_params'])




