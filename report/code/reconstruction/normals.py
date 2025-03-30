import cv2
import numpy as np
from scipy.ndimage import gaussian_filter


def estimate_normals(shading, lighting_params):
    h, w = shading.shape
    light_direction = lighting_params['direction']

    normals = np.zeros((h, w, 3), dtype=np.float32)

    # Оцениваем градиенты
    grad_x = cv2.Sobel(shading, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(shading, cv2.CV_32F, 0, 1, ksize=3)

    # Адаптивный коэффициент
    brightness_std = lighting_params['brightness_std']
    alpha = 0.2 * (1.0 / (brightness_std + 0.05))

    # Расчет нормалей
    normals[:, :, 0] = -alpha * grad_x
    normals[:, :, 1] = -alpha * grad_y
    normals[:, :, 2] = 1.0

    # Сглаживание нормалей
    for c in range(3):
        normals[:, :, c] = gaussian_filter(normals[:, :, c], sigma=0.8)

    # Нормализация
    norm = np.sqrt(np.sum(normals ** 2, axis=2, keepdims=True))
    normals = normals / (norm + 1e-10)

    # Оценка достоверности нормалей
    gradient_magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
    max_magnitude = np.max(gradient_magnitude)
    if max_magnitude > 0:
        gradient_magnitude = gradient_magnitude / max_magnitude

    # Корректировка весов на краях
    edge_correction = 1.0 - gradient_magnitude ** 2
    edge_correction = np.clip(edge_correction, 0.1, 1.0)

    confidence_mask = gradient_magnitude * edge_correction

    # Учет яркости
    brightness_weight = np.clip(shading * 1.5, 0.3, 1.0)
    confidence_mask = confidence_mask * brightness_weight

    # Сглаживание маски
    confidence_mask = gaussian_filter(confidence_mask, sigma=1.5)

    return normals, confidence_mask
