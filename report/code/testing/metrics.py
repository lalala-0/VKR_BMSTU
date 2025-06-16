import numpy as np


def compute_depth_metrics(pred_depth, gt_depth, mask=None):
    """
    Вычисляет метрики оценки качества реконструкции глубины

    Args:
        pred_depth: Предсказанная карта глубины
        gt_depth: Эталонная карта глубины
        mask: Маска действительных пикселей (опционально)
        is_lightfield: Флаг, указывающий, что используется датасет lightfield с инвертированными картами глубины

    Returns:
        dict: Словарь с метриками
    """
    if mask is None:
        mask = gt_depth > 0

    # Копирование для избежания изменения входных данных
    pred = pred_depth[mask].copy()
    gt = gt_depth[mask].copy()

    # Нормализация карт глубины для корректного сравнения
    if len(gt) > 0 and len(pred) > 0:
        # Создаем копии для разных типов нормализации
        pred_median = pred.copy()
        gt_median = gt.copy()
        pred_minmax = pred.copy()
        gt_minmax = gt.copy()

        # Медианное масштабирование для REL и дельт
        scale = np.median(gt_median) / np.median(pred_median)
        pred_median = pred_median * scale

        # Min-Max нормализация для MAE и RMSE
        pred_minmax = (pred_minmax - pred_minmax.min()) / (pred_minmax.max() - pred_minmax.min())
        gt_minmax = (gt_minmax - gt_minmax.min()) / (gt_minmax.max() - gt_minmax.min())

    # Маленькое значение для избежания деления на ноль
    eps = 1e-8

    # Абсолютная средняя ошибка (MAE) - с min-max нормализацией
    mae = np.mean(np.abs(pred_minmax - gt_minmax))

    # Среднеквадратичная ошибка (RMSE) - с min-max нормализацией
    rmse = np.sqrt(np.mean((pred_minmax - gt_minmax) ** 2))

    # Относительная абсолютная ошибка (REL) - с медианным масштабированием
    valid_gt = gt_median > eps
    if np.any(valid_gt):
        rel = np.mean(np.abs(pred_median[valid_gt] - gt_median[valid_gt]) / gt_median[valid_gt])
    else:
        rel = np.nan

    # Точность δ - с медианным масштабированием
    thresh = np.maximum((gt_median / (pred_median + eps)), (pred_median / (gt_median + eps)))
    delta1 = (thresh < 1.25).mean()
    delta2 = (thresh < 1.25 ** 2).mean()
    delta3 = (thresh < 1.25 ** 3).mean()

    return {
        'mae': mae,
        'rmse': rmse,
        'rel': rel,
        'delta1': delta1,
        'delta2': delta2,
        'delta3': delta3
    }


# # Масштабирование предсказания по эталону (rel норм, а mae и rmse - нет)
# scale = np.median(gt) / np.median(pred)
# pred = pred * scale

# # Нормас все, кроме rel. Остальное правдоподобно
# # pred' = a * pred + b
# a = np.cov(gt, pred)[0, 1] / np.var(pred)
# b = np.mean(gt) - a * np.mean(pred)
# pred = a * pred + b

# # Масштабирование по среднему ПЛОХА
# scale = np.mean(gt) / np.mean(pred)
# pred = pred * scale

# # Min-Max нормализация:
# pred = (pred - pred.min()) / (pred.max() - pred.min())
# gt = (gt - gt.min()) / (gt.max() - gt.min())
