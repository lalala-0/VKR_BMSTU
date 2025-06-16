import cv2
import numpy as np
from sklearn.cluster import KMeans

def segment_by_color(img, n_color_segments=6):
    # Работаем с цветным изображением
    if len(img.shape) == 2:
        image_for_cluster = np.stack([img] * 3, axis=2)
    else:
        image_for_cluster = img.copy()

    h, w = img.shape[:2]

    # Уменьшаем размерность данных для ускорения KMeans
    # Для больших изображений делаем downsampling перед кластеризацией
    if h * w > 300000:  # для больших изображений
        scale_factor = np.sqrt(300000 / (h * w))
        small_img = cv2.resize(image_for_cluster, None, fx=scale_factor, fy=scale_factor)
        pixels = small_img.reshape(-1, 3)
    else:
        pixels = image_for_cluster.reshape(-1, 3)

    # Применяем KMeans с меньшим числом iterations и большей tolerance для ускорения
    kmeans = KMeans(n_clusters=n_color_segments, random_state=42,
                    n_init=10, max_iter=100, tol=1e-3)

    # Используем float32 для ускорения вычислений
    pixels = pixels.astype(np.float32)
    labels = kmeans.fit_predict(pixels)

    # Если делали downsampling, возвращаемся к нормальному размеру
    if h * w > 300000:
        full_pixels = image_for_cluster.reshape(-1, 3).astype(np.float32)
        # Предсказываем метки для полного изображения с обученной моделью
        labels = kmeans.predict(full_pixels)

    # Преобразуем обратно в изображение
    segmented = labels.reshape(img.shape[:2])

    # Создаем прямоугольные сегменты вокруг областей каждого цвета
    bboxes = {}

    # Обрабатываем каждый цветовой кластер
    for label in range(n_color_segments):
        # Создаем бинарную маску для текущего цвета
        mask = (segmented == label).astype(np.uint8)

        # Ищем контуры вместо использования regionprops
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for i, contour in enumerate(contours):
            # Получаем ограничивающий прямоугольник напрямую через OpenCV
            x, y, w_rect, h_rect = cv2.boundingRect(contour)

            # Добавляем поля к рамке
            margin = 5
            y1 = max(0, y - margin)
            y2 = min(h, y + h_rect + margin)
            x1 = max(0, x - margin)
            x2 = min(w, x + w_rect + margin)

            # Если область слишком маленькая, пропускаем
            if (y2 - y1) < 20 or (x2 - x1) < 20:
                continue

            # Создаем уникальный идентификатор для этого сегмента
            segment_id = label * 1000 + i + 1

            # Создаем маску для текущего контура
            region_mask = np.zeros_like(mask)
            cv2.drawContours(region_mask, [contour], 0, 1, -1)

            # Оптимизированная версия индексации
            region = region_mask[y1:y2, x1:x2]
            segment_region = segmented[y1:y2, x1:x2]
            segment_region[region > 0] = segment_id

            # Сохраняем ограничивающую рамку
            bboxes[segment_id] = (y1, y2, x1, x2)

    return segmented, bboxes

