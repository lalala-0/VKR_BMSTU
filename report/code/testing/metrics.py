import numpy as np


def compute_depth_metrics(pred_depth, gt_depth, mask=None):
    """
    Вычисляет метрики оценки качества реконструкции глубины

    Args:
        pred_depth: Предсказанная карта глубины
        gt_depth: Эталонная карта глубины
        mask: Маска действительных пикселей (опционально)

    Returns:
        dict: Словарь с метриками
    """
    if mask is None:
        mask = gt_depth > 0

    pred = pred_depth[mask]
    gt = gt_depth[mask]

    # Абсолютная средняя ошибка (MAE)
    mae = np.mean(np.abs(pred - gt)) / 10

    # Среднеквадратичная ошибка (RMSE)
    rmse = np.sqrt(np.mean((pred - gt) ** 2)) / 10

    # Относительная абсолютная ошибка (REL)
    rel = np.mean(np.abs(pred - gt) / gt) / 10

    # Точность δ
    thresh = np.maximum((gt / pred), (pred / gt)) * 10
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


