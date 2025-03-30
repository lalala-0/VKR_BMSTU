import cv2
import numpy as np


def estimate_lighting(image):
    hist = cv2.calcHist([image * 255], [0], None, [256], [0, 256])
    hist_normalized = hist / (image.shape[0] * image.shape[1])
    brightness_mean = np.mean(image)
    brightness_std = np.std(image)

    # Вычисляем градиенты
    grad_x = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=5)
    grad_y = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=5)
    gradient_magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)

    # Определяем области с постоянным освещением
    smooth_regions = gradient_magnitude < np.percentile(gradient_magnitude, 25)

    # Оцениваем направление света по градиентам
    avg_grad_x = np.mean(grad_x)
    avg_grad_y = np.mean(grad_y)

    light_x = -avg_grad_x
    light_y = -avg_grad_y
    light_z = 1.0

    light_direction = np.array([light_x, light_y, light_z])
    norm = np.linalg.norm(light_direction)
    if norm > 0:
        light_direction = light_direction / norm

    lighting_params = {
        'direction': light_direction,
        'brightness_mean': brightness_mean,
        'brightness_std': brightness_std,
        'hist': hist_normalized,
        'smooth_regions': smooth_regions
    }

    return lighting_params
