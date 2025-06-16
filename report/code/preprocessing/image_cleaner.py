import cv2
import numpy as np

def preprocess_archival_photo(image_path, brightness_boost=1.5, contrast_boost=1.2):
    if isinstance(image_path, str):
        print("Загрузка изображения из файла:", image_path)
        image = cv2.imread(image_path)
        if image is not None and len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        print("Использование переданного изображения в формате numpy array")
        image = image_path.copy()

    # Сохраняем оригинальное цветное изображение
    original_colored = image.copy()
    is_colored = len(image.shape) == 3

    # Создаем черно-белую версию для обнаружения артефактов
    if is_colored:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image.copy()

    # Выявление и маскирование дефектов
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    diff = cv2.absdiff(gray, blur)
    _, artifact_mask = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)

    kernel = np.ones((3, 3), np.uint8)
    artifact_mask = cv2.morphologyEx(artifact_mask, cv2.MORPH_CLOSE, kernel)
    artifact_mask = cv2.dilate(artifact_mask, kernel, iterations=1)

    # Инпейнтинг для удаления дефектов (применяем к цветному изображению)
    if is_colored:
        cleaned = cv2.inpaint(original_colored, artifact_mask.astype(np.uint8), 3, cv2.INPAINT_TELEA)
    else:
        cleaned = cv2.inpaint(gray, artifact_mask.astype(np.uint8), 3, cv2.INPAINT_TELEA)

    if is_colored:
        # Для цветного изображения работаем в пространстве LAB
        lab = cv2.cvtColor(cleaned, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)

        # Повышение контраста для канала яркости (L)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_enhanced = clahe.apply(l)

        # Повышение яркости
        mean_brightness = np.mean(l_enhanced)
        target_brightness = 80
        brightness_factor = min((target_brightness / (mean_brightness + 1e-6)), brightness_boost)
        l_brightened = np.clip(l_enhanced * brightness_factor, 0, 255).astype(np.uint8)

        # Улучшение контраста
        alpha = min(contrast_boost, 1.1)
        beta = 128 * (1 - alpha)
        l_enhanced_contrast = np.clip(alpha * l_brightened + beta, 0, 255).astype(np.uint8)

        # Билатеральная фильтрация
        l_denoised = cv2.bilateralFilter(l_enhanced_contrast, 7, 25, 25)

        # Собираем изображение обратно
        lab_enhanced = cv2.merge([l_denoised, a, b])
        enhanced_image = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2RGB)

        # Нормализация до [0, 1]
        preprocessed = enhanced_image.astype(np.float32) / 255.0

        # Гамма-коррекция
        gamma = 1.1
        preprocessed = np.power(preprocessed, gamma)
    else:
        # Для черно-белого изображения используем исходный код
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(cleaned)

        mean_brightness = np.mean(enhanced)
        target_brightness = 80
        brightness_factor = min((target_brightness / (mean_brightness + 1e-6)), brightness_boost)
        brightened = np.clip(enhanced * brightness_factor, 0, 255).astype(np.uint8)

        alpha = min(contrast_boost, 1.1)
        beta = 128 * (1 - alpha)
        enhanced_contrast = np.clip(alpha * brightened + beta, 0, 255).astype(np.uint8)

        denoised = cv2.bilateralFilter(enhanced_contrast, 7, 25, 25)

        preprocessed = denoised.astype(np.float32) / 255.0
        gamma = 1.1
        preprocessed = np.power(preprocessed, gamma)

    return preprocessed, artifact_mask