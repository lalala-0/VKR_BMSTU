import os
import sys
import cv2
import argparse

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
    parser.add_argument('image_path', type=str, nargs='?', default='./img/image.jpg',
                        help='Путь к изображению (по умолчанию: ./img/image.jpg)')
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

    args = parser.parse_args()

    if not os.path.exists(args.image_path):
        print(f"Ошибка: файл {args.image_path} не найден")
        sys.exit(1)

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