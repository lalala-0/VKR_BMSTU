import json
import os
import time

import pandas as pd
from matplotlib import pyplot as plt


def create_result_directories(dataset_name, method):
    """Создание директорий для результатов и визуализаций"""
    results_dir = os.path.join('./results', dataset_name, method)
    os.makedirs(results_dir, exist_ok=True)
    vis_dir = os.path.join(results_dir, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)
    return results_dir, vis_dir

def visualize_depth_results(image, gt_depth, pred_depth, output_path):
    """Визуализация входного изображения, эталонной и предсказанной глубины"""
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
    plt.savefig(output_path)
    plt.close()


def save_metrics_to_json(metrics_avg, all_metrics, results_dir, dataset_name, method, sample_count, processing_time):
    """Сохранение метрик в JSON-файл"""
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(results_dir, f"metrics_{timestamp}.json")

    json_data = {
        'overall_metrics': metrics_avg,
        'per_image_metrics': all_metrics,
        'config': {
            'method': method,
            'dataset': dataset_name,
            'sample_count': sample_count,
            'time': processing_time
        }
    }

    with open(results_file, 'w') as f:
        json.dump(json_data, f, indent=4)

    return results_file


def print_metrics_summary(metrics_avg, dataset_name):
    """Вывод сводки метрик"""
    print(f"\nРезультаты тестирования на датасете {dataset_name}:")
    if 'mae' in metrics_avg:
        print(f"MAE (средняя абсолютная ошибка): {metrics_avg['mae']:.4f}")
    if 'rmse' in metrics_avg:
        print(f"RMSE (среднеквадратичная ошибка): {metrics_avg['rmse']:.4f}")
    if 'rel' in metrics_avg:
        print(f"REL (относительная ошибка): {metrics_avg['rel']:.4f}")
    if 'log10' in metrics_avg:
        print(f"log10 (средняя ошибка log10): {metrics_avg['log10']:.4f}")
    if 'delta1' in metrics_avg:
        print(f"δ1 (% в пределах 1.25): {metrics_avg['delta1'] * 100:.2f}%")
    if 'delta2' in metrics_avg:
        print(f"δ2 (% в пределах 1.25²): {metrics_avg['delta2'] * 100:.2f}%")
    if 'delta3' in metrics_avg:
        print(f"δ3 (% в пределах 1.25³): {metrics_avg['delta3'] * 100:.2f}%")

def save_results_to_csv(comparator):
    """Сохраняет результаты тестирования в CSV формате для удобного анализа"""
    metrics = ['rmse', 'mae', 'rel', 'delta1']

    # Сохраняем результаты по уровням старения
    aging_data = []
    for method in comparator.results:
        for dataset in comparator.results[method]:
            for level in comparator.results[method][dataset]:
                if isinstance(level, (int, float)):
                    for metric in metrics:
                        if metric in comparator.results[method][dataset][level]:
                            aging_data.append({
                                'Метод': method,
                                'Датасет': dataset,
                                'Уровень_старения': level,
                                'Метрика': metric,
                                'Значение': comparator.results[method][dataset][level][metric]
                            })

    if aging_data:
        df_aging = pd.DataFrame(aging_data)
        csv_path = os.path.join(comparator.output_dir, 'aging_levels_results.csv')
        df_aging.to_csv(csv_path, index=False)
        print(f"Результаты по уровням старения сохранены в {csv_path}")

    # Сохраняем результаты по эффектам старения
    effects_data = []
    for method in comparator.results:
        for dataset in comparator.results[method]:
            for key in comparator.results[method][dataset]:
                if isinstance(key, str) and key.startswith('effect_'):
                    effect = key.replace('effect_', '')
                    for intensity, values in comparator.results[method][dataset][key].items():
                        for metric in metrics:
                            if metric in values:
                                effects_data.append({
                                    'Метод': method,
                                    'Датасет': dataset,
                                    'Эффект': effect,
                                    'Интенсивность': float(intensity),
                                    'Метрика': metric,
                                    'Значение': values[metric]
                                })

    if effects_data:
        df_effects = pd.DataFrame(effects_data)
        csv_path = os.path.join(comparator.output_dir, 'aging_effects_results.csv')
        df_effects.to_csv(csv_path, index=False)
        print(f"Результаты по эффектам старения сохранены в {csv_path}")

def plot_aging_effects_comparison(comparator, metric='rmse'):
    """Создает сравнительные графики всех эффектов старения"""
    methods = list(comparator.results.keys())
    datasets = list(set().union(*[comparator.results[method].keys() for method in methods]))

    # Находим все эффекты старения
    all_effects = []
    for method in methods:
        for dataset in datasets:
            if dataset in comparator.results[method]:
                for key in comparator.results[method][dataset].keys():
                    if isinstance(key, str) and key.startswith('effect_'):
                        effect = key.replace('effect_', '')
                        if effect not in all_effects:
                            all_effects.append(effect)

    if not all_effects:
        print("Нет данных для сравнения эффектов старения")
        return

    for dataset in datasets:
        # Сравнение всех эффектов для каждого метода
        for method in methods:
            if dataset not in comparator.results[method]:
                continue

            fig, ax = plt.subplots(figsize=(14, 8))

            for effect in all_effects:
                effect_key = f'effect_{effect}'
                if effect_key in comparator.results[method][dataset]:
                    # Сортируем интенсивности
                    intensities = sorted([float(i) for i in comparator.results[method][dataset][effect_key].keys()])

                    if metric in comparator.results[method][dataset][effect_key][intensities[0]]:
                        values = [comparator.results[method][dataset][effect_key][i][metric] for i in intensities]
                        ax.plot(intensities, values, marker='o', linewidth=2, label=effect)

            ax.set_title(f'Влияние эффектов старения на {metric.upper()} - {method} - {dataset}')
            ax.set_xlabel('Интенсивность эффекта')
            ax.set_ylabel(metric.upper())
            ax.grid(True, alpha=0.3)
            ax.legend()

            plt.tight_layout()
            plt.savefig(os.path.join(comparator.output_dir, f'effects_comparison_{method}_{dataset}_{metric}.png'))
            plt.close()

        # Сравнение устойчивости методов к каждому эффекту
        for effect in all_effects:
            effect_key = f'effect_{effect}'

            fig, ax = plt.subplots(figsize=(14, 8))

            for method in methods:
                if dataset in comparator.results[method] and effect_key in comparator.results[method][dataset]:
                    intensities = sorted([float(i) for i in comparator.results[method][dataset][effect_key].keys()])

                    if metric in comparator.results[method][dataset][effect_key][intensities[0]]:
                        values = [comparator.results[method][dataset][effect_key][i][metric] for i in intensities]
                        # Нормализуем значения относительно исходного для лучшего сравнения
                        baseline = values[0] if values[0] != 0 else 1.0
                        normalized = [v / baseline for v in values]
                        ax.plot(intensities, normalized, marker='o', linewidth=2, label=method)

            ax.set_title(f'Устойчивость методов к эффекту "{effect}" - {dataset}')
            ax.set_xlabel('Интенсивность эффекта')
            ax.set_ylabel(f'Относительное изменение {metric.upper()}')
            ax.grid(True, alpha=0.3)
            ax.legend()

            plt.tight_layout()
            plt.savefig(os.path.join(comparator.output_dir, f'methods_robustness_{effect}_{dataset}_{metric}.png'))
            plt.close()

def generate_summary_report(comparator):
    """Генерирует сводный отчет по результатам сравнения"""
    metrics = ['rmse', 'mae', 'rel']
    report_path = os.path.join(comparator.output_dir, 'summary_report.txt')

    with open(report_path, 'w') as f:
        f.write("=== СВОДНЫЙ ОТЧЕТ ПО СРАВНЕНИЮ МЕТОДОВ РЕКОНСТРУКЦИИ ГЛУБИНЫ ===\n\n")

        # Наилучшие методы по датасетам без старения
        f.write("1. НАИЛУЧШИЕ МЕТОДЫ ПО ДАТАСЕТАМ (БЕЗ ЭФФЕКТОВ СТАРЕНИЯ):\n")
        for metric in metrics:
            f.write(f"\nМетрика: {metric.upper()}\n")

            for dataset in set().union(*[comparator.results[m].keys() for m in comparator.results]):
                f.write(f"  Датасет {dataset}:\n")

                results = []
                for method in comparator.results:
                    if dataset in comparator.results[method] and 0 in comparator.results[method][dataset]:
                        if metric in comparator.results[method][dataset][0]:
                            results.append((method, comparator.results[method][dataset][0][metric]))

                # Сортировка по возрастанию метрики (меньше - лучше)
                results.sort(key=lambda x: x[1])

                for i, (method, value) in enumerate(results):
                    f.write(f"    {i + 1}. {method}: {value:.4f}\n")

        # Устойчивость методов к старению
        f.write("\n2. УСТОЙЧИВОСТЬ МЕТОДОВ К ЭФФЕКТАМ СТАРЕНИЯ:\n")

        for dataset in set().union(*[comparator.results[m].keys() for m in comparator.results]):
            f.write(f"\nДатасет: {dataset}\n")

            for effect in ['sepia', 'vignette', 'noise', 'fading', 'spots']:
                effect_key = f'effect_{effect}'
                f.write(f"  Эффект: {effect}\n")

                results = []
                for method in comparator.results:
                    if dataset in comparator.results[method] and effect_key in comparator.results[method][dataset]:
                        intensities = sorted([float(i) for i in comparator.results[method][dataset][effect_key].keys()])
                        if len(intensities) > 1:
                            # Берем метрику rmse для оценки устойчивости
                            if 'rmse' in comparator.results[method][dataset][effect_key][intensities[0]]:
                                baseline = comparator.results[method][dataset][effect_key][intensities[0]]['rmse']
                                worst = comparator.results[method][dataset][effect_key][intensities[-1]]['rmse']
                                degradation = (worst - baseline) / baseline * 100 if baseline > 0 else float('inf')
                                results.append((method, degradation))

                # Сортировка по возрастанию деградации (меньше - лучше)
                results.sort(key=lambda x: x[1])

                for i, (method, degradation) in enumerate(results):
                    f.write(f"    {i + 1}. {method}: {degradation:.2f}% деградации\n")

    print(f"Сводный отчет сохранен в {report_path}")