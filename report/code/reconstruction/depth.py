import cv2
import numpy as np


def recover_depth(normals, confidence_mask, iterations=200, initial_depth=None):
    h, w = normals.shape[:2]

    # Вычисляем градиенты из нормалей
    nx = normals[:, :, 0]
    ny = normals[:, :, 1]
    nz = normals[:, :, 2] + 1e-10

    p = -nx / nz
    q = -ny / nz

    # Сглаживаем целевые градиенты
    p_smooth = cv2.bilateralFilter(p, 5, 0.1, 5)
    q_smooth = cv2.bilateralFilter(q, 5, 0.1, 5)

    p = p_smooth * confidence_mask + p * (1 - confidence_mask)
    q = q_smooth * confidence_mask + q * (1 - confidence_mask)

    # Инициализация карты глубины
    if initial_depth is not None:
        # Используем предоставленную начальную глубину
        depth = initial_depth.copy()
    else:
        # Создаем новую карту глубины
        depth = np.zeros((h, w), dtype=np.float32)

    # Итеративное уточнение глубины
    for iter in range(iterations):
        depth_x = np.zeros_like(depth)
        depth_y = np.zeros_like(depth)

        depth_x[:, 1:] = depth[:, 1:] - depth[:, :-1]
        depth_y[1:, :] = depth[1:, :] - depth[:-1, :]

        error_x = depth_x - p
        error_y = depth_y - q

        update = np.zeros_like(depth)

        update[:-1, :] += error_y[:-1, :]
        update[:, :-1] += error_x[:, :-1]
        update[1:, :] -= error_y[1:, :]
        update[:, 1:] -= error_x[:, 1:]

        # Применяем обновление с взвешиванием по маске уверенности
        update_weight = 0.5 / iterations
        weighted_update = update * update_weight * confidence_mask
        depth += weighted_update

    # Финальное сглаживание
    depth = cv2.bilateralFilter(depth.astype(np.float32), 9, 75, 75)

    # Нормализация
    p_low, p_high = np.percentile(depth, [2, 98])
    depth = (depth - p_low) / (p_high - p_low + 1e-6)
    depth = np.clip(depth, 0, 1)

    # Гамма-коррекция для усиления деталей
    depth_map = np.power(depth, 0.85)

    depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min() + 1e-6)

    # Инвертирование карты глубины
    depth_map = 1.0 - depth_map

    return depth_map