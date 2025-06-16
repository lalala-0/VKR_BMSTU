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

    def process_segmentation(self, image_array):
        # Предобработка изображения
        try:
            preprocessed, artifact_mask = preprocess_archival_photo(
                image_array,
                self.brightness_boost,
                self.contrast_boost
            )
        except Exception as e:
            print(f"Ошибка при предобработке: {e}")
            return None

        # Получаем цветовые сегменты
        try:
            segments_mask, bboxes = segment_by_color(
                preprocessed,
                self.n_color_segments
            )

        except Exception as e:
            print(f"Ошибка при сегментации: {e}")
            return None
        preprocessed = cv2.cvtColor(preprocessed, cv2.COLOR_RGB2GRAY)

        # Инициализируем карты для хранения результатов
        depth_map = np.zeros_like(preprocessed, dtype=np.float32)
        albedo_map = np.zeros_like(preprocessed, dtype=np.float32)
        shading_map = np.zeros_like(preprocessed, dtype=np.float32)
        normals_map = np.zeros((preprocessed.shape[0], preprocessed.shape[1], 3), dtype=np.float32)

        # Обрабатываем каждый прямоугольный сегмент
        for segment_id, bbox in bboxes.items():
            y1, y2, x1, x2 = bbox

            # Проверка валидности границ
            if y1 >= y2 or x1 >= x2 or y2 > preprocessed.shape[0] or x2 > preprocessed.shape[1]:
                print(f"Пропуск сегмента {segment_id} из-за невалидных границ: {bbox}")
                continue

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
            albedo, shading, smooth_shadow_mask = separate_albedo_shading(segment_img, lighting_params)

            # Проверка размерностей
            if albedo.shape[:2] != segment_mask.shape:
                albedo = cv2.resize(albedo, (segment_mask.shape[1], segment_mask.shape[0]))
            if shading.shape != segment_mask.shape:
                shading = cv2.resize(shading, (segment_mask.shape[1], segment_mask.shape[0]))

            # Оценка нормалей поверхности
            normals, confidence_mask = estimate_normals(shading, lighting_params, smooth_shadow_mask)

            # Проверка размерностей для нормалей
            if normals.shape[:2] != segment_mask.shape:
                normals_resized = np.zeros((segment_mask.shape[0], segment_mask.shape[1], 3), dtype=np.float32)
                for c in range(3):
                    normals_resized[:,:,c] = cv2.resize(normals[:,:,c],
                                                     (segment_mask.shape[1], segment_mask.shape[0]))
                normals = normals_resized

            # Восстановление карты глубины
            segment_depth = recover_depth(normals, confidence_mask, self.iterations)

            # Проверка размерности карты глубины
            if segment_depth.shape != segment_mask.shape:
                segment_depth = cv2.resize(segment_depth, (segment_mask.shape[1], segment_mask.shape[0]))

            # Добавляем результаты в общую карту с учетом маски сегмента
            depth_map[y1:y2, x1:x2] += segment_depth * segment_mask

            # Проверяем размерности для альбедо
            albedo_map[y1:y2, x1:x2] += albedo * np.expand_dims(segment_mask, axis=-1)[:,:,0]
            shading_map[y1:y2, x1:x2] += shading * segment_mask

            # Добавляем нормали
            for c in range(3):
                normals_map[y1:y2, x1:x2, c] += normals[:,:,c] * segment_mask



        # Нормализуем карту глубины
        non_zero_mask = depth_map > 0
        if np.any(non_zero_mask):
            depth_min = depth_map[non_zero_mask].min()
            depth_max = depth_map[non_zero_mask].max()
            depth_range = depth_max - depth_min
            if depth_range > 1e-6:
                depth_map[non_zero_mask] = (depth_map[non_zero_mask] - depth_min) / depth_range
            else:
                depth_map[non_zero_mask] = 0.5  # Устанавливаем среднее значение при маленьком диапазоне

        # Заполняем пробелы в карте глубины
        if np.any(depth_map == 0):
            valid_mask = (depth_map > 0).astype(np.uint8)
            depth_map = cv2.inpaint(depth_map.astype(np.float32),
                                   (valid_mask == 0).astype(np.uint8), 3,
                                   cv2.INPAINT_TELEA)

        try:
            # Постобработка
            final_depth, model_3d = postprocess_depth(depth_map, preprocessed, artifact_mask)
        except Exception as e:
            print(f"Ошибка при постобработке: {e}")
            final_depth = depth_map
            model_3d = None

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
    def process_midas(self, image_array):
        # Предобработка изображения
        preprocessed, artifact_mask = preprocess_archival_photo(
            image_array,
            self.brightness_boost,
            self.contrast_boost
        )
        # # Получаем сегменты для визуализации
        # segments_mask, bboxes = segment_by_color(
        #     preprocessed,
        #     self.n_color_segments
        # )

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
            'albedo': albedo_map,
            'shading': shading_map,
            'normals': normals_map,
            'depth_map': depth_map,
            'final_depth': final_depth,
            'model_3d': model_3d,
        }

        return results

    def process_hybrid(self, image_array):
       """
       Гибридный метод: сплавление карт глубины из MiDaS и сегментационного подхода
       """
       start_time = time.time()
       print("Применяю гибридный метод восстановления глубины...")

       # Шаг 1: Получаем карту глубины с помощью MiDaS
       print("Получение карты глубины через MiDaS...")
       midas_results = self.process_midas(image_array)
       midas_depth = midas_results['final_depth']
       preprocessed = midas_results['preprocessed']

       # Шаг 2: Получаем карту глубины с помощью сегментационного метода
       print("Получение карты глубины через сегментационный метод...")
       segment_results = self.process_segmentation(image_array)
       segment_depth = segment_results['final_depth']

       # Шаг 3: Сплавляем две карты глубины
       print("Интеграция карт глубины...")

       # Получаем веса для сплавления на основе уверенности в деталях сегментации
       # Вычисляем оценку детализации сегментационного метода через градиенты
       detail_weight = np.abs(sobel(segment_depth, axis=0)) + np.abs(sobel(segment_depth, axis=1))
       detail_weight = gaussian_filter(detail_weight, sigma=2.0)

       # Нормализуем веса
       if np.max(detail_weight) > 0:
           detail_weight = detail_weight / np.max(detail_weight)

       # Параметр альфа: базовый вес для карты MiDaS
       alpha_base = 0.9  # Вес базовой карты MiDaS (изменяемый параметр)

       # На участках с высокой детализацией повышаем влияние сегментационного метода
       alpha_map = alpha_base - (detail_weight * 0.4)  # Уменьшаем вес MiDaS там, где есть детали
       alpha_map = np.clip(alpha_map, 0.3, 0.9)  # Ограничиваем диапазон влияния

       # Комбинируем карты глубины
       final_depth = alpha_map * midas_depth + (1 - alpha_map) * segment_depth

       # Постобработка
       artifact_mask = np.zeros((preprocessed.shape[0], preprocessed.shape[1]), dtype=np.uint8)
       final_depth, model_3d = postprocess_depth(final_depth, preprocessed, artifact_mask)

       # Формируем результаты
       results = {
           'preprocessed': preprocessed,
           'segments_mask': segment_results['segments_mask'],
           'albedo': segment_results['albedo'],
           'shading': segment_results['shading'],
           'normals': segment_results['normals'],
           'depth_map': midas_depth,  # Карта от MiDaS
           'segment_depth': segment_depth,  # Карта от сегментационного метода
           'detail_weight': detail_weight,  # Карта весов деталей
           'final_depth': final_depth,  # Сплавленная карта
           'model_3d': model_3d,
           'bboxes': segment_results['bboxes'],
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

    def process_numpy(self, image_array):
        """
        Обрабатывает изображение, переданное в виде NumPy массива.

        Parameters:
        -----------
        image_array : numpy.ndarray
            Изображение в формате numpy array (BGR или RGB)

        Returns:
        --------
        dict
            Словарь с результатами обработки
        """
        start_time = time.time()

        # Обрабатываем временное изображение
        if self.method == "segmentation":
            results = self.process_segmentation(image_array)
        elif self.method == "midas":
            results = self.process_midas(image_array)
        elif self.method == "hybrid":
            results = self.process_hybrid(image_array)
        else:
            raise ValueError(f"Неизвестный метод: {self.method}")

        processing_time = time.time() - start_time
        results['processing_time'] = processing_time
        print(f"Обработка массива NumPy завершена за {processing_time:.2f} секунд")

        return results
