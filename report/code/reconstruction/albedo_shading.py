import numpy as np
from scipy.ndimage import gaussian_filter
import cv2

def separate_albedo_shading(image, lighting_params=None, highlights_mask=None):
    """
    Разделяет изображение на компоненты альбедо и затенения с учётом падающих теней и бликов.

    Args:
        image: входное изображение (одноканальное или многоканальное)
        lighting_params: параметры освещения (словарь)
            - brightness_std: стандартное отклонение яркости
            - direction: направление света [x, y, z]
            - cast_shadow_threshold: порог для детекции падающих теней (по умолчанию 0.15)
            - shadow_softness: мягкость перехода в тенях (по умолчанию 0.05)
            - shadow_strength: сила затемнения теней (по умолчанию 0.7)
            - large_scale_weight: вес крупномасштабного освещения
            - shadow_correction: коэффициент коррекции теней
            - highlight_threshold: порог для детекции бликов (по умолчанию 0.8)
            - highlight_softness: мягкость перехода в бликах (по умолчанию 0.05)
        highlights_mask: маска существующих бликов (опционально)

    Returns:
        tuple: (альбедо, затенение, маска_теней, маска_бликов)
    """
    if lighting_params is None:
        lighting_params = {}

    # Проверка и нормализация входного изображения
    if np.max(image) > 1.0:
        image = image.astype(np.float32) / 255.0

    # Защита от деления на ноль
    epsilon = 1e-6
    image_safe = np.maximum(image, epsilon)

    # Получаем параметры освещения
    brightness_std = lighting_params.get('brightness_std', 0.2)
    light_direction = lighting_params.get('direction', [0, 0, 1])
    cast_shadow_threshold = lighting_params.get('cast_shadow_threshold', 0.15)
    shadow_softness = lighting_params.get('shadow_softness', 0.05)
    shadow_strength = lighting_params.get('shadow_strength', 0.7)
    shadow_correction = lighting_params.get('shadow_correction', 0.3)

    # Определяем многоканальное ли изображение
    is_multichannel = len(image.shape) > 2 and image.shape[2] > 1

    # Если изображение многоканальное, преобразуем его в одноканальное для анализа
    if is_multichannel:
        # Преобразуем в L*a*b* и используем канал яркости
        img_bgr = (image * 255).astype(np.uint8)
        img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
        luminance = img_lab[:, :, 0].astype(np.float32) / 255.0
    else:
        luminance = image_safe.copy()

    # Применяем логарифмическое преобразование
    log_image = np.log1p(luminance)

    # Адаптивные параметры фильтрации на основе освещения
    sigma_large = max(luminance.shape) * 0.09 * (1.0 + 0.5 * brightness_std)
    sigma_small = max(luminance.shape) * 0.09 * (1.0 + 0.3 * brightness_std)

    illumination_large = gaussian_filter(log_image, sigma=sigma_large)
    illumination_small = gaussian_filter(log_image, sigma=sigma_small)

    # Коэффициенты смешивания зависят от яркости
    large_weight = lighting_params.get('large_scale_weight', 0.7 - 0.2 * min(brightness_std, 1.0))
    small_weight = 1.0 - large_weight
    illumination = large_weight * illumination_large + small_weight * illumination_small

    # Вычитаем компоненту освещения
    log_albedo = log_image - illumination

    # Восстанавливаем из логарифмического пространства
    albedo_raw = np.exp(log_albedo) - 1.0

    # Детектирование падающих теней
    # 1. Находим градиент яркости
    dy, dx = np.gradient(luminance)
    gradient_magnitude = np.sqrt(dx**2 + dy**2)

    # 2. Находим резкие переходы (потенциальные границы теней)
    shadow_edges = gradient_magnitude > cast_shadow_threshold

    # 3. Вычисляем маску теней с использованием дилатации
    shadow_edges_dilated = cv2.dilate(
        shadow_edges.astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=2
    )

    # 4. Отфильтровываем маску теней на основе яркости
    shadow_mask = np.logical_and(shadow_edges_dilated, luminance < np.mean(luminance) * 0.8)

    # 5. Сглаживаем маску теней
    shadow_mask = gaussian_filter(shadow_mask.astype(np.float32), sigma=3.0)

    # 6. Применяем сглаживание для создания мягких переходов
    smooth_shadow_mask = gaussian_filter(shadow_mask, sigma=shadow_softness * max(luminance.shape))

    # Корректируем альбедо с учетом теней
    albedo_corrected = albedo_raw.copy()
    shadow_regions = smooth_shadow_mask > 0.2
    if np.any(shadow_regions):
        # Повышаем значения альбедо в теневых областях
        albedo_boost_factor = 1.0 + shadow_correction * smooth_shadow_mask
        albedo_corrected = albedo_raw * albedo_boost_factor

    # Нормализуем альбедо
    p_low, p_high = np.percentile(albedo_corrected, [2, 98])
    albedo = (albedo_corrected - p_low) / (p_high - p_low + epsilon)
    albedo = np.clip(albedo, 0.01, 1.0)

    # Вычисляем затенение
    shading = luminance / (albedo + epsilon)

    # Применяем коррекцию затенения с учетом направления света
    if not np.array_equal(light_direction, [0, 0, 1]):
    # if light_direction != [0, 0, 1]:
        # Создаем координатную сетку
        nx = np.linspace(-1, 1, luminance.shape[1])
        ny = np.linspace(-1, 1, luminance.shape[0])
        x, y = np.meshgrid(nx, ny)

        # Имитация формы освещения на основе направления света
        light_x, light_y, light_z = light_direction
        light_influence = x * light_x + y * light_y + light_z
        light_influence = (light_influence + 1) / 2  # Нормализация до [0, 1]

        # Коррекция затенения с учетом направления света
        shading_correction = np.clip(light_influence, 0.3, 1.0)
        shading = shading * (1.0 + 0.2 * (1.0 - shading_correction))

    # Специальная обработка падающих теней
    shadow_factor = 1.0 - shadow_strength * smooth_shadow_mask
    shading = shading * shadow_factor

    # Нормализуем затенение
    p_low, p_high = np.percentile(shading, [1, 99])
    shading = (shading - p_low) / (p_high - p_low + epsilon)
    shading = np.clip(shading, 0.01, 1.0)

    # Адаптивное сглаживание затенения
    shading_smooth_sigma = 1.0 + 0.5 * brightness_std
    shading = gaussian_filter(shading, sigma=shading_smooth_sigma)


    return albedo, shading, smooth_shadow_mask