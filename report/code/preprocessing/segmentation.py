import cv2
import numpy as np
from skimage.measure import regionprops
from sklearn.cluster import KMeans


def segment_by_color(img, n_color_segments=6):
    # Работаем с цветным изображением
    if len(img.shape) == 2:
        image_for_cluster = np.stack([img] * 3, axis=2)
    else:
        image_for_cluster = img.copy()

    # Преобразуем в формат для кластеризации
    pixels = image_for_cluster.reshape(-1, 3)

    # Применяем KMeans для кластеризации цветов
    kmeans = KMeans(n_clusters=n_color_segments, random_state=42, n_init=10)
    labels = kmeans.fit_predict(pixels)

    # Преобразуем обратно в изображение
    segmented = labels.reshape(img.shape[:2])

    # Создаем прямоугольные сегменты вокруг областей каждого цвета
    h, w = segmented.shape[:2]
    bboxes = {}

    for label in range(n_color_segments):
        # Создаем бинарную маску для текущего цвета
        mask = (segmented == label).astype(np.uint8)

        # Находим все связные компоненты для этого цвета
        num_labels, labels_img = cv2.connectedComponents(mask)

        for region_id in range(1, num_labels):
            region_mask = (labels_img == region_id).astype(np.uint8)

            # Находим прямоугольную ограничивающую рамку
            props = regionprops(region_mask)
            if not props:
                continue

            y1, x1, y2, x2 = props[0].bbox

            # Добавляем поля к рамке
            margin = 5
            y1 = max(0, y1 - margin)
            y2 = min(h, y2 + margin)
            x1 = max(0, x1 - margin)
            x2 = min(w, x2 + margin)

            # Если область слишком маленькая, пропускаем
            if (y2 - y1) < 20 or (x2 - x1) < 20:
                continue

            # Создаем уникальный идентификатор для этого прямоугольного сегмента
            segment_id = label * 1000 + region_id

            # Отмечаем этот сегмент в общей маске
            segmented[y1:y2, x1:x2][region_mask[y1:y2, x1:x2] > 0] = segment_id

            # Сохраняем ограничивающую рамку
            bboxes[segment_id] = (y1, y2, x1, x2)

    return segmented, bboxes