import os
import sys
import cv2
import argparse

from testing.testing import run_testing_nyu, run_testing_scenenet, run_testing_make3d, run_testing_lightfield

# Добавляем путь к корневой директории проекта
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from reconstruction.depth_reconstruction import ArchivalDepthReconstruction
from postprocessing.visualization import visualize_results
from utils.io import save_results, check_and_create_dirs
from compare.compare_without_gt import compare_depth_methods

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
    parser.add_argument('--dataset-type', type=str, choices=['scenenet', 'nyu', 'make3d', 'lightfield'], default='custom',
                        help='Тип датасета: scenenet - sceneNet, nyu - NYU Depth V2, make3d - Make3D, lightfield - 4D Light Field')

    args = parser.parse_args()

    # Режим тестирования
    if args.test:
        if not args.test_dataset:
            print("Ошибка: не указан путь к тестовому датасету (--test-dataset)")
            sys.exit(1)

        if args.dataset_type == 'nyu':
            run_testing_nyu(args)
        elif args.dataset_type == 'scenenet':
            run_testing_scenenet(args)
        elif args.dataset_type == 'make3d':
            run_testing_make3d(args)
        elif args.dataset_type == 'lightfield':
            run_testing_lightfield(args)
        else:
            print(f"Неподдерживаемый тип датасета: {args.dataset_type}")
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

