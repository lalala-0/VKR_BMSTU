# code/compare/compare_without_gt.py
import sys
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import sobel

# Добавляем путь к корневой директории проекта
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from reconstruction.depth_reconstruction import ArchivalDepthReconstruction

def evaluate_depth_without_gt(rgb_image, depth_map):
    """
    Оценка качества карты глубины без эталонных данных
    """
    metrics = {}

    # Преобразуем RGB в оттенки серого
    if len(rgb_image.shape) == 3:
        gray = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)
    else:
        gray = rgb_image.copy()

    # Находим края на исходном изображении
    edges_rgb = cv2.Canny(gray, 50, 150)

    # Нормализуем и преобразуем карту глубины для обнаружения краев
    depth_norm = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min() + 1e-6)
    depth_gray = (depth_norm * 255).astype(np.uint8)
    edges_depth = cv2.Canny(depth_gray, 50, 150)

    # Оцениваем согласованность краев
    edge_overlap = np.logical_and(edges_rgb > 0, edges_depth > 0).sum() / (edges_rgb.sum() + 1e-6)
    metrics['edge_consistency'] = edge_overlap

    # Оцениваем гладкость поверхности
    non_edge_mask = edges_rgb == 0
    if non_edge_mask.sum() > 0:
        depth_variation = np.std(depth_map[non_edge_mask])
        metrics['surface_smoothness'] = 1.0 / (1.0 + depth_variation * 10)  # Масштабируем для удобства

    # Оценка градиентов
    grad_x_rgb = sobel(gray, axis=1)
    grad_y_rgb = sobel(gray, axis=0)
    grad_x_depth = sobel(depth_map, axis=1)
    grad_y_depth = sobel(depth_map, axis=0)

    grad_mag_rgb = np.sqrt(grad_x_rgb**2 + grad_y_rgb**2)
    grad_mag_depth = np.sqrt(grad_x_depth**2 + grad_y_depth**2)

    # Корреляция между градиентами на значимых участках
    mask = grad_mag_rgb > np.percentile(grad_mag_rgb, 70)
    if mask.sum() > 0:
        grad_correlation = np.corrcoef(grad_mag_rgb[mask].flatten(),
                                     grad_mag_depth[mask].flatten())[0, 1]
        metrics['gradient_consistency'] = max(0, grad_correlation)

    # Агрегированная оценка
    if len(metrics) > 0:
        metrics['overall_score'] = np.mean(list(metrics.values()))

    return metrics

def compare_depth_methods(image_path, output_dir="./comparison"):
    """
    Сравнение разных методов оценки глубины

    Args:
        image_path: путь к изображению для сравнения
        output_dir: директория для сохранения результатов

    Returns:
        путь к файлу с визуализацией сравнения или None в случае ошибки
    """
    # Проверяем существование файла
    if not os.path.exists(image_path):
        print(f"Ошибка: файл {image_path} не найден")
        return None

    # Создаем директорию для сохранения результатов
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    try:
        # Загружаем исходное изображение
        rgb_image = cv2.imread(image_path)

        # Проверяем, успешно ли загружено изображение
        if rgb_image is None:
            print(f"Ошибка: не удалось прочитать файл {image_path}. Проверьте формат файла.")
            print("Поддерживаемые форматы: JPEG, PNG, BMP, TIFF и другие форматы, совместимые с OpenCV.")
            return None

        # Создаем экземпляры реконструкторов с разными методами

        reconstructor_seg = ArchivalDepthReconstruction(method="segmentation")
        reconstructor_midas = ArchivalDepthReconstruction(method="midas")
        reconstructor_hybrid = ArchivalDepthReconstruction(method="hybrid")

        methods = {
            "Сегментация": reconstructor_seg,
            "MiDaS": reconstructor_midas,
            "Гибридный": reconstructor_hybrid
        }

        # Загружаем исходное изображение
        rgb_image = cv2.imread(image_path)

        # Получаем карты глубины от разных методов
        depth_maps = {}
        times = {}

        print(f"Сравнение методов для изображения: {image_path}")
        for method_name, reconstructor in methods.items():
            print(f"\nОбработка методом '{method_name}'...")
            results = reconstructor.process(image_path)
            depth_maps[method_name] = results['final_depth']
            times[method_name] = results['processing_time']

        # Оцениваем каждую карту глубины
        evaluations = {}
        for method_name, depth_map in depth_maps.items():
            evaluations[method_name] = evaluate_depth_without_gt(rgb_image, depth_map)

        # Выводим результаты
        print("\n=== Результаты сравнения методов ===")
        print(f"{'Метод':<15} {'Время (с)':<10}")
        print("-" * 65)

        for method_name in methods.keys():
            metrics = evaluations[method_name]
            time_sec = times[method_name]
            edge = metrics.get('edge_consistency', 0)
            smooth = metrics.get('surface_smoothness', 0)
            grad = metrics.get('gradient_consistency', 0)
            overall = metrics.get('overall_score', 0)

            print(f"{method_name:<15} {time_sec:<10.2f}")

        # Визуальное сравнение
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Исходное изображение
        axes[0, 0].imshow(cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB))
        axes[0, 0].set_title("Исходное изображение")
        axes[0, 0].axis('off')

        # Карты глубины
        for i, (method_name, depth_map) in enumerate(depth_maps.items()):
            row, col = (i+1) // 2, (i+1) % 2
            axes[row, col].imshow(depth_map, cmap='plasma')

            metrics = evaluations[method_name]
            title = f"{method_name}\nВремя: {times[method_name]:.2f}с\n"
            title += f"Оценка: {metrics.get('overall_score', 0):.3f}"

            axes[row, col].set_title(title)
            axes[row, col].axis('off')

        plt.tight_layout()
        comparison_path = os.path.join(output_dir, "method_comparison.png")
        plt.savefig(comparison_path)

        # Сохраняем индивидуальные карты глубины
        for method_name, depth_map in depth_maps.items():
            try:
                depth_colored = cv2.applyColorMap((depth_map * 255).astype(np.uint8), cv2.COLORMAP_PLASMA)
                depth_path = os.path.join(output_dir, f"depth_{method_name.lower()}.png")
                cv2.imwrite(depth_path, depth_colored)
            except Exception as e:
                print(f"Ошибка при сохранении карты глубины для метода '{method_name}': {str(e)}")

        print(f"\nРезультаты сравнения сохранены в директорию: {output_dir}")
        return comparison_path

    except Exception as e:
        print(f"Произошла непредвиденная ошибка: {str(e)}")
        return None