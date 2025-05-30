import os
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from collections import defaultdict
from testing.testing import run_testing
from matplotlib.ticker import MaxNLocator

from testing.inout import save_results_to_csv, plot_aging_effects_comparison


class ModelComparator:
    """Класс для сравнения методов реконструкции глубины"""

    def __init__(self, output_dir="./research_results"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.results = defaultdict(lambda: defaultdict(dict))

    def run_comparisons(self, methods, datasets, aging_levels):
        """Запуск тестирования для всех комбинаций параметров"""
        for method in methods:
            for dataset in datasets:
                for aging_level in aging_levels:
                    print(f"\n{'=' * 80}")
                    print(f"Тестирование: метод={method}, датасет={dataset}, уровень старения={aging_level}")
                    print(f"{'=' * 80}")

                    # Создаем объект аргументов для тестирования
                    class Args:
                        pass

                    args = Args()
                    args.method = method
                    args.dataset_type = dataset
                    args.test_dataset = self._get_dataset_path(dataset)
                    args.max_samples = 20  # Ограничиваем для ускорения
                    args.batch_size = 1
                    args.workers = 2


                    # Настраиваем эффекты старения
                    args.aging_effects = aging_level > 0
                    if aging_level == 0:  # Без деградации
                        args.sepia_intensity = 0.0
                        args.vignette_enabled = False
                        args.noise_intensity = 0.0
                        args.fading_intensity = 0.0
                        args.spots_intensity = 0.0
                    else: # Определяем общее количество уровней (кроме нулевого)
                        total_aging_levels = max(aging_levels)
                        # Определяем, к какой категории относится текущий уровень
                        # и вычисляем нормализованную позицию внутри категории
                        if total_aging_levels == 1:
                            # Если есть только один уровень старения
                            alpha = 0.5  # Средняя интенсивность
                        else:
                            # Линейная интерполяция между минимумом и максимумом
                            alpha = (aging_level - 1) / (total_aging_levels - 1)

                        # Параметры сепии: от 0.3 до 1.0
                        args.sepia_intensity = 0.3 + 0.7 * alpha
                        # Параметры виньетирования
                        args.vignette_enabled = True
                        args.vignette_radius = 0.7 - 0.2 * alpha  # от 0.7 до 0.5
                        args.vignette_intensity = 0.3 + 0.5 * alpha  # от 0.3 до 0.8
                        # Параметры шума: от 0.1 до 0.5
                        args.noise_intensity = 0.0
                        # Параметры выцветания: от 0.3 до 0.8
                        args.fading_intensity = 0.3 + 0.5 * alpha
                        # Параметры пятен: от 0.05 до 0.3
                        args.spots_intensity = 0.05 + 0.25 * alpha

                    # Запускаем тестирование
                    metrics = run_testing(args)

                    if metrics:
                        self.results[method][dataset][aging_level] = metrics

                    # Сохраняем промежуточные результаты
                    self._save_results()

        # Визуализация результатов после завершения всех тестов
        print(f"\n{'#' * 80}")
        print("Сохранение и визуализация результатов тестирования")
        print(f"{'#' * 80}")

        # Сохраняем результаты в CSV формате
        save_results_to_csv(self)

        # Строим графики с аналитикой по всем метрикам
        self.plot_all_metrics()

    def _get_dataset_path(self, dataset):
        """Получение пути к датасету"""
        dataset_paths = {
            'nyu': './testing/datasets/nyu/nyu_depth_v2_labeled.mat',
            'make3d': './testing/datasets/make3d',
            'scenenet': './testing/datasets/sceneNet/scenenet_rgbd',
            'lightfield': './testing/datasets/lightfield'
        }
        return dataset_paths.get(dataset, './datasets')

    def _save_results(self):
        """Сохранение результатов в JSON-файл"""
        results_file = os.path.join(self.output_dir, 'comparison_results.json')

        # Рекурсивная функция для преобразования структур данных
        def convert_for_json(obj):
            if isinstance(obj, defaultdict):
                return {k: convert_for_json(v) for k, v in obj.items()}
            elif isinstance(obj, dict):
                return {str(k): convert_for_json(v) for k, v in obj.items()}
            elif isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, (np.int32, np.int64)):
                return int(obj)
            else:
                return obj

        # Применяем преобразование к результатам
        results_dict = convert_for_json(self.results)

        # Создаем директорию, если её нет
        os.makedirs(os.path.dirname(results_file), exist_ok=True)

        with open(results_file, 'w') as f:
            json.dump(results_dict, f, indent=4)

        print(f"Результаты сохранены в {results_file}")

    def load_results(self, file_path=None):
        """Загрузка результатов из JSON-файла"""
        if file_path is None:
            file_path = os.path.join(self.output_dir, 'comparison_results.json')

        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                loaded = json.load(f)

            # Преобразуем обратно в наш формат
            for method, datasets in loaded.items():
                for dataset, levels in datasets.items():
                    for level, metrics in levels.items():
                        self.results[method][dataset][int(level)] = metrics
            return True
        return False

    def plot_metrics_by_dataset(self, metric='rmse'):
        """Построение графика метрики по датасетам для всех методов"""
        methods = list(self.results.keys())
        datasets = list(set().union(*[self.results[method].keys() for method in methods]))

        x = np.arange(len(datasets))
        if len(methods) == 0:
            width = 1
        else:
            width = 0.35 / len(methods)

        fig, ax = plt.subplots(figsize=(12, 8))

        for i, method in enumerate(methods):
            values = []
            for dataset in datasets:
                if dataset in self.results[method] and 0 in self.results[method][dataset]:
                    values.append(self.results[method][dataset][0][metric])
                else:
                    values.append(0)

            ax.bar(x + i * width - width * len(methods) / 2, values, width, label=method)

        ax.set_title(f'Сравнение метрики {metric.upper()} по датасетам')
        ax.set_xlabel('Датасет')
        ax.set_ylabel(metric.upper())
        ax.set_xticks(x)
        ax.set_xticklabels(datasets)
        ax.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f'metrics_by_dataset_{metric}.png'))
        plt.close()

    def plot_metrics_by_aging(self, metric='rmse'):
        """Построение графика метрики в зависимости от уровня старения для всех методов и датасетов"""
        from scipy.interpolate import make_interp_spline

        methods = list(self.results.keys())
        datasets = list(set().union(*[self.results[method].keys() for method in methods]))

        for dataset in datasets:
            fig, ax = plt.subplots(figsize=(12, 8))

            for method in methods:
                if dataset in self.results[method]:
                    aging_levels = sorted(self.results[method][dataset].keys())
                    values = [self.results[method][dataset][level][metric] for level in aging_levels]

                    # Добавляем сглаживание 2 порядка если точек достаточно
                    if len(aging_levels) >= 3:
                        # Создаем новые точки для гладкой кривой
                        x_smooth = np.linspace(min(aging_levels), max(aging_levels), 100)
                        try:
                            # Создаем сплайн 2 порядка
                            spl = make_interp_spline(aging_levels, values, k=2)
                            y_smooth = spl(x_smooth)
                            # Рисуем сглаженную линию
                            ax.plot(x_smooth, y_smooth, linewidth=2, label=f"{method}")
                            # Добавляем оригинальные точки
                            ax.scatter(aging_levels, values, marker='o')
                        except:
                            # Если не удалось создать сплайн, используем обычный график
                            ax.plot(aging_levels, values, marker='o', linewidth=2, label=method)
                    else:
                        # Если точек недостаточно для сплайна 2 порядка, используем обычный график
                        ax.plot(aging_levels, values, marker='o', linewidth=2, label=method)

            ax.set_title(f'{metric.upper()} по уровням старения - датасет {dataset}')
            ax.set_xlabel('Уровень старения')
            ax.set_ylabel(metric.upper())
            ax.grid(True)
            ax.legend()
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))

            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, f'metrics_by_aging_{dataset}_{metric}.png'))
            plt.close()
    def plot_heatmap_comparison(self, metric='rmse'):
        """Построение тепловой карты для сравнения методов, датасетов и уровней старения"""
        data = []

        for method in self.results:
            for dataset in self.results[method]:
                for level in self.results[method][dataset]:
                    data.append({
                        'Метод': method,
                        'Датасет': dataset,
                        'Уровень старения': level,
                        metric.upper(): self.results[method][dataset][level][metric]
                    })

        if not data:
            return

        df = pd.DataFrame(data)

        # Создаем сводную таблицу
        pivot = df.pivot_table(
            values=metric.upper(),
            index=['Метод', 'Датасет'],
            columns='Уровень старения',
            aggfunc='mean'
        )

        fig, ax = plt.subplots(figsize=(12, 10))
        im = ax.imshow(pivot.values, cmap='coolwarm')

        # Настройка осей
        ax.set_xticks(np.arange(len(pivot.columns)))
        ax.set_yticks(np.arange(len(pivot.index)))
        ax.set_xticklabels(pivot.columns)
        ax.set_yticklabels(pivot.index)

        # Подписи значений в ячейках
        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                text = ax.text(j, i, f'{pivot.values[i, j]:.3f}',
                               ha="center", va="center", color="black")

        plt.colorbar(im, ax=ax)
        ax.set_title(f'Тепловая карта метрики {metric.upper()} по методам, датасетам и уровням старения')
        fig.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f'heatmap_{metric}.png'))
        plt.close()


    def run_aging_effect_comparison(self, methods, dataset, effect_type, intensities):
        """
        Запуск тестирования для конкретного эффекта старения с разными интенсивностями

        Parameters:
        -----------
        methods : list
            Список методов для сравнения
        dataset : str
            Датасет для тестирования
        effect_type : str
            Тип эффекта старения ('sepia', 'vignette', 'noise', 'fading', 'spots')
        intensities : list
            Список значений интенсивности для тестирования (от 0.0 до 1.0)
        """
        results_key = f"effect_{effect_type}"

        for method in methods:
            for intensity in intensities:
                print(f"\n{'=' * 80}")
                print(
                    f"Тестирование: метод={method}, датасет={dataset}, эффект={effect_type}, интенсивность={intensity}")
                print(f"{'=' * 80}")

                # Создаем объект аргументов для тестирования
                class Args:
                    pass

                args = Args()
                args.method = method
                args.dataset_type = dataset
                args.test_dataset = self._get_dataset_path(dataset)
                args.max_samples = 3  # Ограничиваем для ускорения
                args.batch_size = 1
                args.workers = 2

                # Отключаем все эффекты старения
                args.aging_effects = True
                args.sepia_intensity = 0.0
                args.vignette_enabled = False
                args.vignette_radius = 0.7
                args.vignette_intensity = 0.0
                args.noise_intensity = 0.0
                args.fading_intensity = 0.0
                args.spots_intensity = 0.0

                # Включаем только тестируемый эффект
                if effect_type == 'sepia':
                    args.sepia_intensity = intensity
                elif effect_type == 'vignette':
                    args.vignette_enabled = True
                    args.vignette_intensity = intensity
                elif effect_type == 'noise':
                    args.noise_intensity = intensity
                elif effect_type == 'fading':
                    args.fading_intensity = intensity
                elif effect_type == 'spots':
                    args.spots_intensity = intensity

                # Запускаем тестирование
                metrics = run_testing(args)

                if metrics:
                    if results_key not in self.results[method][dataset]:
                        self.results[method][dataset][results_key] = {}
                    self.results[method][dataset][results_key][intensity] = metrics

                # Сохраняем промежуточные результаты
                self._save_results()

    def plot_metrics_by_aging_effect(self, metric='rmse'):
        """Построение графика метрики в зависимости от интенсивности эффекта старения"""
        from scipy.interpolate import make_interp_spline

        methods = list(self.results.keys())
        datasets = list(set().union(*[self.results[method].keys() for method in methods]))

        # Ищем все эффекты старения, для которых есть данные
        all_effects = []
        for method in methods:
            for dataset in datasets:
                if dataset in self.results[method]:
                    for key in self.results[method][dataset].keys():
                        if isinstance(key, str) and key.startswith('effect_'):
                            effect = key.replace('effect_', '')
                            if effect not in all_effects:
                                all_effects.append(effect)

        # Для каждого эффекта строим график
        for effect in all_effects:
            effect_key = f'effect_{effect}'

            for dataset in datasets:
                fig, ax = plt.subplots(figsize=(12, 8))

                for method in methods:
                    if dataset in self.results[method] and effect_key in self.results[method][dataset]:
                        # Сортируем интенсивности
                        intensities = sorted([float(i) for i in self.results[method][dataset][effect_key].keys()])
                        values = [self.results[method][dataset][effect_key][i][metric] for i in intensities]

                        # Добавляем сглаживание 2 порядка если точек достаточно
                        if len(intensities) >= 3:
                            # Создаем новые точки для гладкой кривой
                            x_smooth = np.linspace(min(intensities), max(intensities), 100)
                            try:
                                # Создаем сплайн 2 порядка
                                spl = make_interp_spline(intensities, values, k=2)
                                y_smooth = spl(x_smooth)
                                # Рисуем сглаженную линию
                                ax.plot(x_smooth, y_smooth, linewidth=2, label=f"{method}")
                                # Добавляем оригинальные точки
                                ax.scatter(intensities, values, marker='o')
                            except:
                                # Если не удалось создать сплайн, используем обычный график
                                ax.plot(intensities, values, marker='o', linewidth=2, label=method)
                        else:
                            # Если точек недостаточно для сплайна 2 порядка, используем обычный график
                            ax.plot(intensities, values, marker='o', linewidth=2, label=method)

                ax.set_title(f'{metric.upper()} при изменении интенсивности эффекта "{effect}" - датасет {dataset}')
                ax.set_xlabel(f'Интенсивность эффекта {effect}')
                ax.set_ylabel(metric.upper())
                ax.grid(True)
                ax.legend()

                plt.tight_layout()
                plt.savefig(os.path.join(self.output_dir, f'metrics_by_{effect}_{dataset}_{metric}.png'))
                plt.close()

    def plot_all_metrics(self):
        """Построение всех графиков для всех метрик"""
        metrics = ['mae', 'rmse', 'rel']
        if any('mae' in self.results[m][d][l] for m in self.results for d in self.results[m] for l in
               self.results[m][d] if isinstance(l, (int, float))):
            metrics.append('mae')

        for metric in metrics:
            print(f"Построение графиков для метрики {metric}...")
            self.plot_metrics_by_dataset(metric)
            self.plot_metrics_by_aging(metric)
            self.plot_heatmap_comparison(metric)
            self.plot_metrics_by_aging_effect(metric)

    def run_comprehensive_aging_comparison(self, methods, dataset):
        """
        Запускает комплексное сравнение всех эффектов старения

        Parameters:
        -----------
        methods : list
            Список методов для сравнения
        dataset : str
            Датасет для тестирования
        """
        # Тестируем каждый эффект старения отдельно
        effects = ['sepia', 'vignette', 'noise', 'fading', 'spots']
        intensities = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]  # Разные уровни интенсивности

        for effect in effects:
            print(f"\nТестирование эффекта старения: {effect}")
            self.run_aging_effect_comparison(methods, dataset, effect, intensities)

        # Сохраняем результаты в формате CSV
        save_results_to_csv(self)
        # Строим сравнительные графики
        plot_aging_effects_comparison(self)
        # Строим графики с аналитикой по всем метрикам
        self.plot_all_metrics()

    def plot_results(self, results_file=None):
        """
        Загружает результаты из файла и строит все графики

        Parameters:
        -----------
        results_file : str, optional
            Путь к файлу с результатами (по умолчанию - стандартный путь)
        """
        # Загружаем результаты, если они еще не загружены
        if not self.results or results_file is not None:
            if results_file is not None and results_file.endswith('.csv'):
                loaded = self.load_results_from_csv(results_file)
            else:
                loaded = self.load_results(results_file)

            if not loaded:
                print(f"Ошибка: не удалось загрузить результаты из файла")
                return
            print(f"Результаты успешно загружены")

        # Проверяем, есть ли данные по эффектам старения
        effects_exist = False
        for method in self.results:
            for dataset in self.results[method]:
                for key in self.results[method][dataset]:
                    if isinstance(key, str) and key.startswith('effect_'):
                        effects_exist = True
                        break


        # Если есть данные по эффектам старения, строим дополнительные графики
        if effects_exist:
            print("Построение графиков для эффектов старения...")
            plot_aging_effects_comparison(self)
        else:
            print("Построение графиков для всех метрик...")
            self.plot_all_metrics()

        print(f"Все графики сохранены в директорию {self.output_dir}")

    def load_results_from_csv(self, file_path):
        """Загрузка результатов из CSV-файла"""
        try:
            import pandas as pd

            # Проверяем, существует ли файл
            if not os.path.exists(file_path):
                print(f"Файл {file_path} не найден")
                return False

            # Загружаем CSV в DataFrame
            df = pd.read_csv(file_path)

            # Определяем формат CSV на основе заголовков
            if all(col in df.columns for col in ['Метод', 'Датасет', 'Эффект', 'Интенсивность', 'Метрика', 'Значение']):
                # Формат файла эффектов старения
                for _, row in df.iterrows():
                    method = row['Метод']
                    dataset = row['Датасет']
                    effect = row['Эффект']
                    intensity = float(row['Интенсивность'])
                    metric = row['Метрика'].lower()
                    value = float(row['Значение'])

                    # Формируем ключ эффекта и добавляем данные
                    effect_key = f"effect_{effect}"
                    if effect_key not in self.results[method][dataset]:
                        self.results[method][dataset][effect_key] = {}

                    if intensity not in self.results[method][dataset][effect_key]:
                        self.results[method][dataset][effect_key][intensity] = {}

                    self.results[method][dataset][effect_key][intensity][metric] = value

            elif all(col in df.columns for col in ['Метод', 'Датасет', 'Уровень_старения', 'Метрика', 'Значение']):
                # Формат файла уровней старения
                for _, row in df.iterrows():
                    method = row['Метод']
                    dataset = row['Датасет']
                    aging_level = int(row['Уровень_старения'])
                    metric = row['Метрика'].lower()
                    value = float(row['Значение'])

                    if aging_level not in self.results[method][dataset]:
                        self.results[method][dataset][aging_level] = {}

                    self.results[method][dataset][aging_level][metric] = value

            else:
                print("Неподдерживаемый формат CSV файла. Ожидаемые форматы:")
                print("1. Метод,Датасет,Эффект,Интенсивность,Метрика,Значение")
                print("2. Метод,Датасет,Уровень_старения,Метрика,Значение")
                return False

            print(f"Данные успешно загружены из {file_path}")
            return True

        except Exception as e:
            print(f"Ошибка при загрузке данных из CSV: {str(e)}")
            return False