import torch
from .metrics import compute_depth_metrics

def test_depth_reconstruction_method(model, test_loader):
    """
    Тестирует модель реконструкции глубины на подготовленном наборе данных

    Args:
        model: Модель реконструкции глубины
        test_loader: DataLoader с тестовыми данными

    Returns:
        dict: Метрики по всему набору данных
    """
    metrics_sum = {'mae': 0, 'rmse': 0, 'rel': 0,
                   'delta1': 0, 'delta2': 0, 'delta3': 0}
    sample_count = 0

    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            images = batch['image']
            gt_depth = batch['depth']

            # Предсказание глубины
            pred_depth = model(images)

            # Приведение к одинаковому формату
            pred_depth = pred_depth.cpu().numpy()
            gt_depth = gt_depth.cpu().numpy()

            # Вычисление метрик для батча
            batch_metrics = compute_depth_metrics(pred_depth, gt_depth)

            # Накопление метрик
            for k, v in batch_metrics.items():
                metrics_sum[k] += v * images.size(0)
            sample_count += images.size(0)

    # Усреднение метрик
    metrics_avg = {k: v / sample_count for k, v in metrics_sum.items()}

    return metrics_avg


