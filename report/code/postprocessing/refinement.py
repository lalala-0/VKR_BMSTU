import cv2
import numpy as np
from scipy.ndimage import gaussian_filter


def postprocess_depth(depth_map, preprocessed, artifact_mask):
    preprocessed_img = preprocessed
    if len(preprocessed.shape) == 2:
        preprocessed_img = np.stack([preprocessed] * 3, axis=2)

    # Выделение краев для сохранения структуры
    edges = cv2.Canny(np.uint8(preprocessed * 255), 30, 120)
    edges = edges / 255.0

    kernel = np.ones((3, 3), np.uint8)
    edges_dilated = cv2.dilate(edges, kernel, iterations=1)

    edges_smooth = gaussian_filter(edges_dilated, sigma=1.0)

    # Билатеральная фильтрация для сохранения краев
    depth_smooth = cv2.bilateralFilter(depth_map.astype(np.float32), 9, 50, 50)

    # Комбинирование сглаженной глубины с исходной на краях
    edge_weight = edges_smooth * 1.2
    edge_weight = np.clip(edge_weight, 0, 1)
    final_depth = depth_smooth * (1.0 - edge_weight) + depth_map * edge_weight

    # Коррекция артефактов
    if artifact_mask is not None:
        artifact_weight = artifact_mask.astype(np.float32) / 255.0
        artifact_weight = gaussian_filter(artifact_weight, sigma=2.0)

        neighbor_depth = gaussian_filter(final_depth, sigma=5.0)
        final_depth = final_depth * (1.0 - artifact_weight) + neighbor_depth * artifact_weight

    # Нормализация
    p_low, p_high = np.percentile(final_depth, [1, 99])
    final_depth = (final_depth - p_low) / (p_high - p_low + 1e-6)
    final_depth = np.clip(final_depth, 0, 1)

    # Усиление деталей
    blurred = gaussian_filter(final_depth, sigma=2.0)
    detail_mask = final_depth - blurred
    final_depth = final_depth + detail_mask * 0.5
    final_depth = np.clip(final_depth, 0, 1)

    # Генерация 3D модели
    h, w = final_depth.shape
    y, x = np.mgrid[0:h, 0:w]

    x = (x - w / 2) / max(w, h)
    y = (y - h / 2) / max(w, h)

    z = final_depth * 0.5

    points = np.stack((x.flatten(), y.flatten(), z.flatten()), axis=1)

    if len(preprocessed.shape) == 2:
        colors = np.stack([preprocessed] * 3, axis=2).reshape(-1, 3)
    else:
        colors = preprocessed_img.reshape(-1, 3)

    model_3d = {
        'points': points,
        'colors': colors,
        'shape': (h, w)
    }

    return final_depth, model_3d
