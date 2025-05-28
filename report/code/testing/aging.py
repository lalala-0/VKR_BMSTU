import cv2
import numpy as np


def apply_sepia(image, intensity=0.5):
    """
    Применение эффекта сепии с регулируемой интенсивностью

    Args:
        image: Исходное изображение
        intensity: Интенсивность эффекта (0.0 - без эффекта, 1.0 - максимальный эффект)

    Returns:
        Изображение с примененным эффектом сепии
    """
    # Исходное цветное изображение
    original = image.copy()

    # Применяем преобразование сепии
    sepia_kernel = np.array([
        [0.393, 0.769, 0.189],
        [0.349, 0.686, 0.168],
        [0.272, 0.534, 0.131]
    ])
    sepia_image = cv2.transform(image, sepia_kernel)
    sepia_image = np.clip(sepia_image, 0, 255).astype(np.uint8)

    # Смешиваем с оригинальным изображением в зависимости от интенсивности
    return cv2.addWeighted(original, 1 - intensity, sepia_image, intensity, 0)


def apply_vignette(image, radius=0.7, intensity=0.5):
    """
    Применение эффекта виньетирования с регулируемыми параметрами

    Args:
        image: Исходное изображение
        radius: Радиус виньетки (относительно размера изображения, 0.5-0.7)
        intensity: Интенсивность эффекта (0.0 - без эффекта, 1.0 - максимальный эффект)

    Returns:
        Изображение с примененным эффектом виньетирования
    """
    # Исходное изображение
    original = image.copy()

    # Создаем маску виньетирования
    rows, cols = image.shape[:2]
    # Центр изображения
    center_x = cols / 2
    center_y = rows / 2

    # Создаем радиальную маску
    Y, X = np.ogrid[:rows, :cols]
    dist_from_center = np.sqrt((X - center_x) ** 2 + (Y - center_y) ** 2)

    # Нормализуем расстояние
    max_dist = np.sqrt(center_x ** 2 + center_y ** 2)
    norm_dist = dist_from_center / max_dist

    # Создаем маску виньетирования
    mask = 1 - np.clip(norm_dist - radius, 0, 1) / (1 - radius)
    mask = mask ** 2  # Более плавное затухание

    # Применяем маску к каждому каналу
    vignette = image.copy()
    for c in range(3):
        vignette[:, :, c] = image[:, :, c] * mask

    # Смешиваем с оригинальным изображением в зависимости от интенсивности
    return cv2.addWeighted(original, 1 - intensity, vignette, intensity, 0)


def apply_noise(image, intensity=0.2):
    """
    Добавление шума на изображение

    Args:
        image: Исходное изображение
        intensity: Интенсивность шума (0.0 - без шума, 1.0 - сильный шум)

    Returns:
        Изображение с добавленным шумом
    """
    # Исходное изображение
    original = image.copy()

    # Создаем случайный шум
    noise = np.zeros_like(image, dtype=np.float32)
    cv2.randn(noise, 0, 255 * intensity)

    # Добавляем шум к изображению
    noisy_image = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    return noisy_image


def apply_fading(image, intensity=0.5):
    """
    Применение эффекта выцветания

    Args:
        image: Исходное изображение
        intensity: Интенсивность эффекта (0.0 - без эффекта, 1.0 - максимальный эффект)

    Returns:
        Изображение с эффектом выцветания
    """
    # Исходное изображение
    original = image.copy()

    # Уменьшаем насыщенность (конвертируем в HSV, уменьшаем S, возвращаем в BGR)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] *= (1.0 - intensity)  # Уменьшаем насыщенность
    hsv[:, :, 2] = hsv[:, :, 2] * (1.0 - intensity * 0.5) + 255 * intensity * 0.5  # Увеличиваем яркость
    faded = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    return faded


def apply_spots(image, intensity=0.1):
    """
    Добавление случайных пятен на изображение

    Args:
        image: Исходное изображение
        intensity: Интенсивность эффекта (0.0 - без эффекта, 1.0 - максимальный эффект)

    Returns:
        Изображение с добавленными пятнами
    """
    # Исходное изображение
    result = image.copy()
    rows, cols = image.shape[:2]

    # Количество пятен зависит от интенсивности и размера изображения
    num_spots = int(intensity * 50)  # От 0 до ~50 пятен

    for _ in range(num_spots):
        # Случайная позиция пятна
        cx = np.random.randint(0, cols)
        cy = np.random.randint(0, rows)

        # Случайный радиус пятна
        radius = np.random.randint(5, 20)

        # Случайный цвет пятна (желтоватый/коричневый оттенок для эффекта старения)
        color = np.random.randint(100, 200)
        spot_color = (np.random.randint(0, 50), np.random.randint(80, 150), np.random.randint(150, 230))

        # Создаем маску для пятна с плавными краями
        mask = np.zeros((rows, cols), dtype=np.float32)
        cv2.circle(mask, (cx, cy), radius, 1.0, -1)

        # Размытие маски для более плавных краев
        mask = cv2.GaussianBlur(mask, (radius * 2 + 1, radius * 2 + 1), 0)

        # Применяем пятно к каждому каналу
        for c in range(3):
            result[:, :, c] = np.where(
                mask > 0,
                result[:, :, c] * (1 - mask * intensity) + spot_color[c] * mask * intensity,
                result[:, :, c]
            )

    return result.astype(np.uint8)


def apply_aging_effects(image, args):
    """
    Применение всех эффектов старения к изображению в соответствии с параметрами

    Args:
        image: Исходное изображение
        args: Аргументы с параметрами эффектов
            - aging_effects: включить/выключить все эффекты
            - sepia_intensity: интенсивность эффекта сепии
            - vignette_enabled: включить/выключить виньетирование
            - vignette_radius: радиус виньетирования
            - vignette_intensity: интенсивность виньетирования
            - noise_intensity: интенсивность шума
            - fading_intensity: интенсивность выцветания
            - spots_intensity: интенсивность пятен

    Returns:
        Изображение с примененными эффектами старения
    """
    if not args.aging_effects:
        return image

    result = image.copy()

    # Применяем эффект сепии
    if hasattr(args, 'sepia_intensity') and args.sepia_intensity > 0:
        result = apply_sepia(result, args.sepia_intensity)

    # Применяем эффект виньетирования
    if hasattr(args, 'vignette_enabled') and args.vignette_enabled:
        vignette_radius = getattr(args, 'vignette_radius', 0.7)
        vignette_intensity = getattr(args, 'vignette_intensity', 0.5)
        result = apply_vignette(result, vignette_radius, vignette_intensity)

    # Применяем шум
    if hasattr(args, 'noise_intensity') and args.noise_intensity > 0:
        result = apply_noise(result, args.noise_intensity)

    # Применяем эффект выцветания
    if hasattr(args, 'fading_intensity') and args.fading_intensity > 0:
        result = apply_fading(result, args.fading_intensity)

    # Добавляем пятна
    if hasattr(args, 'spots_intensity') and args.spots_intensity > 0:
        result = apply_spots(result, args.spots_intensity)

    return result