import argparse

from testing.inout import plot_aging_effects_comparison
from testing.compare_methods import ModelComparator


def main():

    parser = argparse.ArgumentParser(description='Сравнение методов восстановления глубины')
    parser.add_argument('--methods', nargs='+', default=['hybrid', 'regression', 'optimization'],
                        help='Методы для сравнения')
    parser.add_argument('--datasets', nargs='+', default=['nyu', 'make3d', 'scenenet', 'lightfield'],
                        help='Датасеты для тестирования')
    parser.add_argument('--aging-levels', type=int, default=3,
                        help='Количество уровней старения (0 - без старения)')
    parser.add_argument('--output-dir', default='./comparison_results',
                        help='Директория для сохранения результатов')
    parser.add_argument('--load-only', action='store_true',
                        help='Только загрузить существующие результаты и построить графики')
    parser.add_argument('--comprehensive-aging', action='store_true',
                        help='Запустить комплексное сравнение эффектов старения')
    parser.add_argument('--aging-dataset', default='nyu',
                        help='Датасет для комплексного тестирования эффектов старения')

    args = parser.parse_args()

    comparator = ModelComparator(args.output_dir)

    if not args.load_only:
        # Если указан флаг комплексного тестирования эффектов старения
        if args.comprehensive_aging:
            print(f"\n{'#' * 80}")
            print(f"Запуск комплексного сравнения эффектов старения для датасета {args.aging_dataset}")
            print(f"{'#' * 80}")
            comparator.run_comprehensive_aging_comparison(args.methods, args.aging_dataset)
        else:
            # Запускаем тестирование для всех комбинаций
            aging_levels = list(range(args.aging_levels + 1))  # 0, 1, 2, ..., args.aging_levels
            comparator.run_comparisons(args.methods, args.datasets, aging_levels)



if __name__ == '__main__':
    main()