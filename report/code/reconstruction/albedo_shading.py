import numpy as np
from scipy.ndimage import gaussian_filter


def separate_albedo_shading(image, lighting_params):
    # Применяем логарифмическое преобразование
    log_image = np.log1p(image)

    # Используем разные размеры ядра Гаусса для улучшения разделения
    sigma_large = max(image.shape) * 0.08
    sigma_small = max(image.shape) * 0.02

    illumination_large = gaussian_filter(log_image, sigma=sigma_large)
    illumination_small = gaussian_filter(log_image, sigma=sigma_small)

    illumination = 0.7 * illumination_large + 0.3 * illumination_small

    # Вычитаем компоненту освещения
    log_albedo = log_image - illumination

    # Восстанавливаем из логарифмического пространства
    albedo = np.exp(log_albedo) - 1.0

    # Ограничиваем значения
    albedo = np.clip(albedo, 0.01, 1.0)

    # Нормализуем альбедо
    p_low, p_high = np.percentile(albedo, [2, 98])
    albedo_norm = (albedo - p_low) / (p_high - p_low + 1e-6)
    albedo = np.clip(albedo_norm, 0.01, 1.0)

    # Вычисляем затенение
    shading = image / (albedo + 1e-6)

    # Ограничиваем и нормализуем затенение
    shading = np.clip(shading, 0.01, 1.0)

    p_low, p_high = np.percentile(shading, [2, 98])
    shading = (shading - p_low) / (p_high - p_low + 1e-6)
    shading = np.clip(shading, 0.01, 1.0)

    # Сглаживаем затенение
    shading = gaussian_filter(shading, sigma=1.0)

    return albedo, shading
