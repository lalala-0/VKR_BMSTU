import os
import sys
import numpy as np
import cv2
import time

from scipy.ndimage import gaussian_filter, sobel

# Добавляем путь к директории code
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Изменим импорты, чтобы они указывали на директории внутри code
from preprocessing.image_cleaner import preprocess_archival_photo
from preprocessing.segmentation import segment_by_color
from reconstruction.lighting import estimate_lighting
from reconstruction.albedo_shading import separate_albedo_shading
from reconstruction.normals import estimate_normals
from reconstruction.depth import recover_depth
from reconstruction.midas_depth import estimate_depth_midas
from postprocessing.refinement import postprocess_depth

class ArchivalDepthReconstruction:
    def __init__(self, iterations=200, lambda_smooth=0.15, regularization_weight=0.3,
                 brightness_boost=1.5, contrast_boost=1.2, n_color_segments=6,
                 method="segmentation", midas_model="MiDaS_small"):
        self.iterations = iterations
        self.lambda_smooth = lambda_smooth
        self.regularization_weight = regularization_weight
        self.brightness_boost = brightness_boost
        self.contrast_boost = contrast_boost
        self.n_color_segments = n_color_segments
        self.method = method
        self.midas_model = midas_model

    def process_segmentation(self, image_path):
        # Предобработка изображения
        preprocessed, artifact_mask = preprocess_archival_photo(
            image_path,
            self.brightness_boost,
            self.contrast_boost
        )

        # Получаем цветовые сегменты
        segments_mask, bboxes = segment_by_color(
            preprocessed,
            self.n_color_segments
        )

        # Инициализируем карты для хранения результатов
        depth_map = np.zeros_like(preprocessed)
        albedo_map = np.zeros_like(preprocessed)
        shading_map = np.zeros_like(preprocessed)
        normals_map = np.zeros((preprocessed.shape[0], preprocessed.shape[1], 3), dtype=np.float32)

        # Обрабатываем каждый прямоугольный сегмент
        for segment_id, bbox in bboxes.items():
            y1, y2, x1, x2 = bbox

            # Создаем маску для текущего сегмента
            segment_mask = (segments_mask[y1:y2, x1:x2] == segment_id).astype(np.float32)

            # Если сегмент слишком маленький, пропускаем
            if np.sum(segment_mask) < 100:
                continue

            # Обрабатываем сегмент
            segment_img = preprocessed[y1:y2, x1:x2].copy()

            # Оценка условий освещения
            lighting_params = estimate_lighting(segment_img)

            # Разделение альбедо и затенения
            albedo, shading = separate_albedo_shading(segment_img, lighting_params)

            # Оценка нормалей поверхности
            normals, confidence_mask = estimate_normals(shading, lighting_params)

            # Восстановление карты глубины
            segment_depth = recover_depth(normals, confidence_mask, self.iterations)

            # Добавляем результаты в общую карту с учетом маски сегмента
            depth_map[y1:y2, x1:x2] += segment_depth * segment_mask
            albedo_map[y1:y2, x1:x2] += albedo * segment_mask
            shading_map[y1:y2, x1:x2] += shading * segment_mask

            # Добавляем нормали
            for c in range(3):
                normals_map[y1:y2, x1:x2, c] += normals[:, :, c] * segment_mask

        # Нормализуем карту глубины
        non_zero_mask = depth_map > 0
        if np.any(non_zero_mask):
            depth_map[non_zero_mask] = (depth_map[non_zero_mask] - depth_map[non_zero_mask].min()) / \
                                       (depth_map[non_zero_mask].max() - depth_map[non_zero_mask].min() + 1e-6)

        # Заполняем пробелы в карте глубины
        if np.any(depth_map == 0):
            valid_mask = (depth_map > 0).astype(np.uint8)
            depth_map = cv2.inpaint(depth_map.astype(np.float32), (valid_mask == 0).astype(np.uint8), 3,
                                    cv2.INPAINT_TELEA)

        # Постобработка
        final_depth, model_3d = postprocess_depth(depth_map, preprocessed, artifact_mask)

        # Формируем результаты
        results = {
            'preprocessed': preprocessed,
            'segments_mask': segments_mask,
            'albedo': albedo_map,
            'shading': shading_map,
            'normals': normals_map,
            'depth_map': depth_map,
            'final_depth': final_depth,
            'model_3d': model_3d,
            'bboxes': bboxes
        }

        return results

    def process_midas(self, image_path):
        # Предобработка изображения
        preprocessed, artifact_mask = preprocess_archival_photo(
            image_path,
            self.brightness_boost,
            self.contrast_boost
        )

        # Получаем сегменты для визуализации
        segments_mask, bboxes = segment_by_color(
            preprocessed,
            self.n_color_segments
        )

        # Оценка глубины с помощью MiDaS
        print(f"Восстановление глубины с помощью MiDaS ({self.midas_model})...")
        depth_map = estimate_depth_midas(preprocessed, model_type=self.midas_model)

        # Постобработка
        final_depth, model_3d = postprocess_depth(depth_map, preprocessed, artifact_mask)

        # Нулевые заполнители для остальных карт
        h, w = preprocessed.shape[:2]
        albedo_map = np.zeros_like(preprocessed)
        shading_map = np.zeros_like(preprocessed)
        normals_map = np.zeros((h, w, 3), dtype=np.float32)

        # Формируем результаты
        results = {
            'preprocessed': preprocessed,
            'segments_mask': segments_mask,
            'albedo': albedo_map,
            'shading': shading_map,
            'normals': normals_map,
            'depth_map': depth_map,
            'final_depth': final_depth,
            'model_3d': model_3d,
            'bboxes': bboxes
        }

        return results

    def process_hybrid(self, image_path):
        """
        Гибридный метод: основная карта глубин из MiDaS + детализация через сегментацию
        """
        start_time = time.time()
        print("Применяю гибридный метод восстановления глубины...")

        # Шаг 1: Получаем базовую карту глубины с помощью MiDaS
        print("Получение базовой карты глубины через MiDaS...")
        midas_results = self.process_midas(image_path)
        base_depth = midas_results['depth_map']
        preprocessed = midas_results['preprocessed']
        artifact_mask = np.zeros_like(preprocessed, dtype=np.uint8)

        # Шаг 2: Получаем сегменты для детализации
        print("Сегментация изображения для детализации...")
        segments_mask, bboxes = segment_by_color(
            preprocessed,
            self.n_color_segments
        )

        # Шаг 3: Инициализируем карты для деталей через сегментационный метод
        detail_depth = np.zeros_like(base_depth)
        detail_weight = np.zeros_like(base_depth)

        # Шаг 4: Обрабатываем каждый сегмент для получения деталей
        print("Детализация каждого сегмента...")
        for segment_id, bbox in bboxes.items():
            y1, y2, x1, x2 = bbox

            # Создаем маску для текущего сегмента
            segment_mask = (segments_mask[y1:y2, x1:x2] == segment_id).astype(np.float32)

            # Пропускаем слишком маленькие сегменты
            if np.sum(segment_mask) < 100:
                continue

            # Получаем подизображение и соответствующую часть базовой глубины
            segment_img = preprocessed[y1:y2, x1:x2].copy()
            segment_base_depth = base_depth[y1:y2, x1:x2].copy()

            # Оценка условий освещения
            lighting_params = estimate_lighting(segment_img)

            # Разделение альбедо и затенения
            albedo, shading = separate_albedo_shading(segment_img, lighting_params)

            # Оценка нормалей поверхности
            normals, confidence_mask = estimate_normals(shading, lighting_params)

            # Модификация: используем базовую глубину как начальное приближение
            # и восстанавливаем только локальные детали
            segment_details = recover_depth(
                normals,
                confidence_mask,
                iterations=self.iterations // 2,  # Меньше итераций
                initial_depth=segment_base_depth  # Используем MiDaS как инициализацию
            )

            # Вычисляем вес деталей на основе уверенности и контраста
            segment_weight = confidence_mask * np.abs(sobel(segment_img, axis=0)) * np.abs(sobel(segment_img, axis=1))
            segment_weight = gaussian_filter(segment_weight, sigma=1.0)

            # Нормализуем вес
            if np.max(segment_weight) > 0:
                segment_weight = segment_weight / np.max(segment_weight)

            # Добавляем результаты с учетом веса
            detail_depth[y1:y2, x1:x2] += segment_details * segment_mask * segment_weight
            detail_weight[y1:y2, x1:x2] += segment_mask * segment_weight

        # Шаг 5: Комбинируем базовую карту и детали
        # Заменяем ноль на очень маленькое число, чтобы избежать деления на ноль
        detail_weight = np.maximum(detail_weight, 1e-6)

        # Нормализуем детали
        detail_depth = detail_depth / detail_weight

        # Адаптивное смешивание базовой карты и деталей
        # Параметр альфа: высокий вес для базовой карты, но на участках с высокой уверенностью
        # в деталях повышаем их влияние
        alpha_base = 0.7  # Вес базовой карты MiDaS (изменяемый параметр)
        alpha_map = alpha_base - (detail_weight * 0.4)  # Уменьшаем вес MiDaS там, где есть хорошие детали
        alpha_map = np.clip(alpha_map, 0.3, 0.9)  # Ограничиваем диапазон влияния

        # Комбинируем карты глубины
        final_depth = alpha_map * base_depth + (1 - alpha_map) * detail_depth

        # Постобработка
        final_depth, model_3d = postprocess_depth(final_depth, preprocessed, artifact_mask)

        # Формируем результаты
        results = {
            'preprocessed': preprocessed,
            'segments_mask': segments_mask,
            'albedo': midas_results['albedo'],  # Используем из MiDaS для простоты
            'shading': midas_results['shading'],
            'normals': midas_results['normals'],
            'depth_map': base_depth,  # Исходная карта от MiDaS
            'detail_depth': detail_depth,  # Карта деталей
            'detail_weight': detail_weight,  # Карта весов деталей
            'final_depth': final_depth,  # Комбинированная карта
            'model_3d': model_3d,
            'bboxes': bboxes,
            'processing_time': time.time() - start_time
        }

        return results

    def process(self, image_path):
        start_time = time.time()

        if self.method == "segmentation":
            results = self.process_segmentation(image_path)
        elif self.method == "midas":
            results = self.process_midas(image_path)
        elif self.method == "hybrid":
            results = self.process_hybrid(image_path)
        else:
            raise ValueError(f"Неизвестный метод: {self.method}")

        processing_time = time.time() - start_time
        results['processing_time'] = processing_time
        print(f"Обработка завершена за {processing_time:.2f} секунд")

        return results