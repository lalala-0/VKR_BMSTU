import os
import sys
import cv2
import argparse
import numpy as np
import torch


# Добавляем путь к корневой директории проекта
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from reconstruction.depth_reconstruction import ArchivalDepthReconstruction
from postprocessing.visualization import visualize_results
from utils.io import save_results, check_and_create_dirs
from compare.compare_without_gt import compare_depth_methods


def run_testing_eth3d(args):
    """
    Тестирование модели на датасете ETH3D

    Args:
        args: Аргументы командной строки
    """
    import numpy as np
    import os
    import cv2
    import json
    import matplotlib.pyplot as plt
    import time
    import glob
    from testing.metrics import compute_depth_metrics

    # Создаем директорию для результатов
    results_dir = os.path.join('./results', 'eth3d', args.method)
    os.makedirs(results_dir, exist_ok=True)

    # Определяем путь к данным ETH3D
    eth3d_base_dir = args.test_dataset
    scenes = args.scenes.split(',') if args.scenes else None

    # Если не указаны конкретные сцены, используем все доступные
    if not scenes:
        # Ищем все доступные сцены (папки, содержащие изображения и карты глубины)
        scenes = [d for d in os.listdir(eth3d_base_dir)
                  if os.path.isdir(os.path.join(eth3d_base_dir, d)) and
                  os.path.exists(os.path.join(eth3d_base_dir, d, "dslr_undistorted")) and
                  os.path.exists(os.path.join(eth3d_base_dir, d, "dslr_depth"))]

    print(f"Найдено {len(scenes)} сцен для тестирования: {', '.join(scenes)}")

    # Создаем модель реконструкции
    from reconstruction.depth_reconstruction import ArchivalDepthReconstruction

    reconstructor = ArchivalDepthReconstruction(
        iterations=args.iterations if hasattr(args, 'iterations') else 200,
        lambda_smooth=0.2,
        brightness_boost=args.brightness if hasattr(args, 'brightness') else 1.8,
        contrast_boost=args.contrast if hasattr(args, 'contrast') else 1.5,
        n_color_segments=args.segments if hasattr(args, 'segments') else 8,
        method=args.method,
        midas_model=args.midas_model if hasattr(args, 'midas_model') else 'MiDaS_small'
    )

    # Метрики
    all_metrics = []
    metrics_sum = {'mae': 0, 'rmse': 0, 'rel': 0, 'delta1': 0, 'delta2': 0, 'delta3': 0}
    total_images = 0

    # Директория для визуализаций
    vis_dir = os.path.join(results_dir, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)

    # Обрабатываем каждую сцену
    start_time = time.time()
    for scene in scenes:
        print(f"\nОбработка сцены {scene}...")

        # Пути к изображениям и картам глубины для текущей сцены
        img_dir = os.path.join(eth3d_base_dir, scene, "dslr_undistorted")
        depth_dir = os.path.join(eth3d_base_dir, scene, "dslr_depth")

        # Список файлов изображений
        image_files = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))

        # Ограничиваем количество образцов, если нужно
        max_samples = args.max_samples if hasattr(args, 'max_samples') and args.max_samples else len(image_files)
        image_files = image_files[:min(max_samples, len(image_files))]

        print(f"Будет обработано {len(image_files)} изображений из {scene}")

        # Создаем поддиректорию для визуализаций сцены
        scene_vis_dir = os.path.join(vis_dir, scene)
        os.makedirs(scene_vis_dir, exist_ok=True)

        # Обрабатываем каждое изображение
        for i, img_path in enumerate(image_files):
            # Получаем соответствующий файл глубины
            base_name = os.path.basename(img_path).split('.')[0]
            depth_path = os.path.join(depth_dir, f"{base_name}.png")  # или другой формат

            if not os.path.exists(depth_path):
                print(f"Предупреждение: карта глубины не найдена для {img_path}")
                continue

            try:
                # Загружаем изображение и карту глубины
                image = cv2.imread(img_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                # Загружаем карту глубины (формат зависит от датасета)
                gt_depth = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH).astype(np.float32)

                # Обрабатываем изображение
                results = reconstructor.process(img_path)
                pred_depth = results.get('final_depth', results.get('depth_map'))

                # Приводим к одинаковым размерам
                if pred_depth.shape != gt_depth.shape:
                    pred_depth = cv2.resize(pred_depth, (gt_depth.shape[1], gt_depth.shape[0]))

                # Вычисляем метрики для этого изображения
                img_metrics = compute_depth_metrics(
                    pred_depth[np.newaxis, ...],
                    gt_depth[np.newaxis, ...]
                )

                # Сохраняем метрики
                img_data = {k: float(v) for k, v in img_metrics.items()}
                img_data['scene'] = scene
                img_data['image_id'] = base_name
                all_metrics.append(img_data)

                # Накапливаем суммарные метрики
                for k, v in img_metrics.items():
                    metrics_sum[k] += v

                # Визуализация (каждые 5 изображений или согласно настройкам)
                if i % 5 == 0 or i == len(image_files) - 1:
                    plt.figure(figsize=(15, 5))

                    plt.subplot(1, 3, 1)
                    plt.imshow(image)
                    plt.title('Входное изображение')
                    plt.axis('off')

                    plt.subplot(1, 3, 2)
                    plt.imshow(gt_depth, cmap='plasma')
                    plt.title('Эталонная глубина')
                    plt.axis('off')

                    plt.subplot(1, 3, 3)
                    plt.imshow(pred_depth, cmap='plasma')
                    plt.title('Предсказанная глубина')
                    plt.axis('off')

                    plt.tight_layout()
                    plt.savefig(os.path.join(scene_vis_dir, f"{base_name}.png"))
                    plt.close()

                total_images += 1
                print(f"Обработано изображение {i + 1}/{len(image_files)}, MAE: {img_metrics['mae']:.4f}")

            except Exception as e:
                print(f"Ошибка при обработке изображения {img_path}: {str(e)}")

    # Вычисляем средние метрики
    if total_images > 0:
        metrics_avg = {k: v / total_images for k, v in metrics_sum.items()}

        # Выводим результаты
        print("\nРезультаты тестирования:")
        print(f"Всего обработано изображений: {total_images}")
        print(f"MAE (средняя абсолютная ошибка): {metrics_avg['mae']:.4f}")
        print(f"RMSE (среднеквадратичная ошибка): {metrics_avg['rmse']:.4f}")
        print(f"REL (относительная ошибка): {metrics_avg['rel']:.4f}")
        print(f"δ1 (% в пределах 1.25): {metrics_avg['delta1'] * 100:.2f}%")
        print(f"δ2 (% в пределах 1.25²): {metrics_avg['delta2'] * 100:.2f}%")
        print(f"δ3 (% в пределах 1.25³): {metrics_avg['delta3'] * 100:.2f}%")

        # Сохраняем результаты в файл
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        results_file = os.path.join(results_dir, f"metrics_{timestamp}.json")

        json_data = {
            'overall_metrics': metrics_avg,
            'per_image_metrics': all_metrics,
            'config': {
                'method': args.method,
                'dataset': 'eth3d',
                'scenes': scenes,
                'total_images': total_images,
                'time': time.time() - start_time
            }
        }

        with open(results_file, 'w') as f:
            json.dump(json_data, f, indent=4)

        print(f"Результаты сохранены в {results_file}")
        return metrics_avg
    else:
        print("Ошибка: не удалось обработать ни одного изображения.")
        return None

def run_testing_nyu(args):
    """
    Улучшенное тестирование модели на датасете NYU

    Args:
        args: Аргументы командной строки
    """
    import h5py
    import numpy as np
    import os
    import cv2
    import json
    import matplotlib.pyplot as plt
    import time
    from testing.metrics import compute_depth_metrics

    # Создаем директорию для результатов
    results_dir = os.path.join('./results', 'nyu', args.method)
    os.makedirs(results_dir, exist_ok=True)

    # Загружаем данные вручную, минуя DataLoader
    print(f"Загрузка NYU Depth Dataset V2 из {args.test_dataset}...")
    data = h5py.File(args.test_dataset, 'r')

    # Получаем массивы изображений и глубины
    images = np.array(data['images'])
    depths = np.array(data['depths'])
    data.close()  # Закрываем файл h5py сразу после загрузки

    # Ограничиваем количество образцов
    max_samples = 100 #args.max_samples if hasattr(args, 'max_samples') and args.max_samples else len(images)
    sample_count = min(max_samples, len(images))
    print(f"Загружено {len(images)} образцов из NYU Depth Dataset V2")
    print(f"Будет обработано {sample_count} образцов")

    # Создаем модель реконструкции
    from reconstruction.depth_reconstruction import ArchivalDepthReconstruction

    reconstructor = ArchivalDepthReconstruction(
        iterations=args.iterations if hasattr(args, 'iterations') else 200,
        lambda_smooth=0.2,
        brightness_boost=args.brightness if hasattr(args, 'brightness') else 1.8,
        contrast_boost=args.contrast if hasattr(args, 'contrast') else 1.5,
        n_color_segments=args.segments if hasattr(args, 'segments') else 8,
        method=args.method,
        midas_model=args.midas_model if hasattr(args, 'midas_model') else 'MiDaS_small'
    )

    # Метрики
    all_metrics = []
    metrics_sum = {'mae': 0, 'rmse': 0, 'rel': 0, 'delta1': 0, 'delta2': 0, 'delta3': 0}

    # Тестирование
    print(f"Запуск тестирования метода {args.method} на датасете NYU...")
    start_time = time.time()

    # Директория для визуализаций
    vis_dir = os.path.join(results_dir, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)

    # Обрабатываем каждое изображение отдельно
    for i in range(sample_count):
        # Преобразуем формат изображения
        image = np.transpose(images[i], (1, 2, 0))  # [C,H,W] -> [H,W,C]
        gt_depth = depths[i]

        # Сохраняем во временный файл
        temp_path = os.path.join(results_dir, f"temp_img_{i}.jpg")
        cv2.imwrite(temp_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

        try:
            # Обрабатываем изображение
            results = reconstructor.process(temp_path)
            pred_depth = results.get('final_depth', results.get('depth_map'))

            # Приводим к одинаковым размерам
            if pred_depth.shape != gt_depth.shape:
                pred_depth = cv2.resize(pred_depth, (gt_depth.shape[1], gt_depth.shape[0]))

            # Вычисляем метрики для этого изображения
            img_metrics = compute_depth_metrics(
                pred_depth[np.newaxis, ...],
                gt_depth[np.newaxis, ...]
            )

            # Сохраняем метрики
            img_data = {k: float(v) for k, v in img_metrics.items()}
            img_data['image_id'] = i
            all_metrics.append(img_data)

            # Накапливаем суммарные метрики
            for k, v in img_metrics.items():
                metrics_sum[k] += v

            # Визуализация (каждые 10 изображений или настраиваемо)
            if i % 10 == 0 or i == sample_count - 1:
                plt.figure(figsize=(15, 5))

                plt.subplot(1, 3, 1)
                plt.imshow(image)
                plt.title('Входное изображение')
                plt.axis('off')

                plt.subplot(1, 3, 2)
                plt.imshow(gt_depth, cmap='plasma')
                plt.title('Эталонная глубина')
                plt.axis('off')

                plt.subplot(1, 3, 3)
                plt.imshow(pred_depth, cmap='plasma')
                plt.title('Предсказанная глубина')
                plt.axis('off')

                plt.tight_layout()
                plt.savefig(os.path.join(vis_dir, f"sample_{i}.png"))
                plt.close()

            print(f"Обработано изображение {i + 1}/{sample_count}, MAE: {img_metrics['mae']:.4f}")

        except Exception as e:
            print(f"Ошибка при обработке изображения {i}: {str(e)}")

        # Удаляем временный файл
        if os.path.exists(temp_path):
            os.remove(temp_path)

    # Вычисляем средние метрики
    metrics_avg = {k: v / sample_count for k, v in metrics_sum.items()}

    # Выводим результаты
    print("\nРезультаты тестирования:")
    print(f"MAE (средняя абсолютная ошибка): {metrics_avg['mae']:.4f}")
    print(f"RMSE (среднеквадратичная ошибка): {metrics_avg['rmse']:.4f}")
    print(f"REL (относительная ошибка): {metrics_avg['rel']:.4f}")
    print(f"δ1 (% в пределах 1.25): {metrics_avg['delta1'] * 100:.2f}%")
    print(f"δ2 (% в пределах 1.25²): {metrics_avg['delta2'] * 100:.2f}%")
    print(f"δ3 (% в пределах 1.25³): {metrics_avg['delta3'] * 100:.2f}%")

    # Сохраняем результаты в файл
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(results_dir, f"metrics_{timestamp}.json")

    json_data = {
        'overall_metrics': metrics_avg,
        'per_image_metrics': all_metrics,
        'config': {
            'method': args.method,
            'dataset': 'nyu',
            'sample_count': sample_count,
            'time': time.time() - start_time
        }
    }

    with open(results_file, 'w') as f:
        json.dump(json_data, f, indent=4)

    print(f"Результаты сохранены в {results_file}")
    return metrics_avg

def visualize_test_samples(model, test_loader, output_dir, num_samples=5):
    """
    Визуализирует несколько примеров предсказаний глубины

    Args:
        model: Модель реконструкции глубины
        test_loader: DataLoader с тестовыми данными
        output_dir: Директория для сохранения результатов
        num_samples: Количество примеров для визуализации
    """
    import matplotlib.pyplot as plt
    import numpy as np

    samples_dir = os.path.join(output_dir, "samples")
    os.makedirs(samples_dir, exist_ok=True)

    # Получение нескольких примеров из датасета
    samples = []
    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            images = batch['image']
            gt_depth = batch['depth']

            # Предсказание глубины
            pred_depth = model(images)

            # Сохраняем примеры
            for i in range(images.size(0)):
                if len(samples) >= num_samples:
                    break

                samples.append({
                    'image': images[i].cpu().numpy(),
                    'gt_depth': gt_depth[i].cpu().numpy(),
                    'pred_depth': pred_depth[i].cpu().numpy()
                })

            if len(samples) >= num_samples:
                break

    # Визуализация примеров
    for i, sample in enumerate(samples):
        plt.figure(figsize=(15, 5))

        plt.subplot(1, 3, 1)
        plt.imshow(np.transpose(sample['image'], (1, 2, 0)))
        plt.title('Входное изображение')
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.imshow(sample['gt_depth'], cmap='plasma')
        plt.title('Эталонная глубина')
        plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.imshow(sample['pred_depth'], cmap='plasma')
        plt.title('Предсказанная глубина')
        plt.axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join(samples_dir, f"sample_{i + 1}.png"))
        plt.close()

    print(f"Визуализированные примеры сохранены в {samples_dir}")

def main():
    # Проверяем и создаем директории по умолчанию
    check_and_create_dirs({
        './img': 'для входных изображений',
        './out': 'для выходных результатов'
    })

    # Обработка аргументов командной строки
    parser = argparse.ArgumentParser(description='Восстановление глубины из архивных фотографий')
    parser.add_argument('--image_path', type=str, nargs='?', default='./img/image.jpg',
                        help='Путь к изображению (по умолчанию: ./img/testimg6.png)')
    parser.add_argument('--output', '-o', type=str, default='./out',
                        help='Директория для сохранения результатов (по умолчанию: ./out)')
    parser.add_argument('--no-display', '-nd', action='store_true', help='Не показывать результаты')
    parser.add_argument('--segments', '-s', type=int, default=8, help='Количество цветовых сегментов')
    parser.add_argument('--iterations', '-i', type=int, default=200,
                        help='Количество итераций для восстановления глубины')
    parser.add_argument('--brightness', '-b', type=float, default=1.8, help='Коэффициент усиления яркости')
    parser.add_argument('--contrast', '-c', type=float, default=1.5, help='Коэффициент усиления контраста')
    parser.add_argument('--method', '-m', type=str, default='segmentation',
                        choices=['segmentation', 'midas', 'hybrid'],
                        help='Метод восстановления глубины')
    parser.add_argument('--midas-model', type=str, default='MiDaS_small',
                        choices=['MiDaS_small', 'DPT_Large', 'DPT_Hybrid'],
                        help='Модель MiDaS для оценки глубины')
    parser.add_argument('--compare', action='store_true',
                        help='Запустить сравнение всех методов')

    # Обновленная часть с аргументами
    parser.add_argument('--test', action='store_true',
                        help='Запустить тестирование на датасете с эталонами')
    parser.add_argument('--test-dataset', type=str,
                        help='Путь к директории с тестовым датасетом или файлу .mat для NYU')
    parser.add_argument('--batch-size', type=int, default=4,
                        help='Размер батча для тестирования')
    parser.add_argument('--workers', type=int, default=2,
                        help='Количество параллельных процессов для загрузки данных')
    parser.add_argument('--aging-effects', action='store_true',
                        help='Применять эффекты старения к тестовым изображениям')
    parser.add_argument('--sepia', action='store_true',
                        help='Применять эффект сепии к тестовым изображениям')
    parser.add_argument('--vignette', action='store_true',
                        help='Применять эффект виньетирования к тестовым изображениям')
    parser.add_argument('--max-samples', type=int,
                        help='Максимальное количество образцов для тестирования')
    parser.add_argument('--dataset-type', type=str, choices=['eth3d', 'nyu', 'kitti'], default='custom',
                        help='Тип датасета: eth3d - eth3d, nyu - NYU Depth V2, kitti - KITTI')
    parser.add_argument('--split', type=str, choices=['train', 'test'], default='train',
                        help='Использовать тренировочный или тестовый сплит датасета')

    args = parser.parse_args()

    # Режим тестирования
    if args.test:
        if not args.test_dataset:
            print("Ошибка: не указан путь к тестовому датасету (--test-dataset)")
            sys.exit(1)

        if args.dataset_type == 'nyu':
            run_testing_nyu(args)
        elif args.dataset_type == 'eth3d':
            run_testing_eth3d(args)
        sys.exit(0)

    # Режим сравнения методов
    if args.compare:
        comparison_path = compare_depth_methods(args.image_path, os.path.join(args.output, "comparison"))

        # Показываем результаты, если требуется
        if not args.no_display:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(12, 10))
            comparison_img = plt.imread(comparison_path)
            plt.imshow(comparison_img)
            plt.axis('off')
            plt.title("Сравнение методов восстановления глубины")
            plt.tight_layout()
            plt.show()

        sys.exit(0)

    if not os.path.exists(args.image_path):
        print(f"Ошибка: файл {args.image_path} не найден")
        sys.exit(1)

    # Создаем экземпляр реконструктора с указанными параметрами
    reconstructor = ArchivalDepthReconstruction(
        iterations=args.iterations,
        lambda_smooth=0.2,
        brightness_boost=args.brightness,
        contrast_boost=args.contrast,
        n_color_segments=args.segments,
        method=args.method,
        midas_model=args.midas_model
    )

    # Обрабатываем изображение
    print(f"Обработка изображения {args.image_path}...")
    print(f"Используемый метод: {args.method}")
    results = reconstructor.process(args.image_path)

    # Визуализируем результаты
    visualization = visualize_results(results)

    # Показываем результаты, если требуется
    if not args.no_display:
        cv2.imshow("Original", visualization['preprocessed'])
        cv2.imshow("Depth Colored", visualization['depth_colored'])
        cv2.imshow("Relief", visualization['relief_colored'])
        if args.method in ['segmentation', 'hybrid']:
            cv2.imshow("Segments", visualization['segments'])

        print(f"Время обработки: {results['processing_time']:.2f} секунд")
        print("Нажмите любую клавишу для закрытия окон...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # Сохраняем результаты
    save_results(visualization, args.output)




if __name__ == "__main__":
    main()

