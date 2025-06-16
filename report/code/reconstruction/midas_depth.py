import cv2
import torch
import numpy as np
import warnings

# Подавление предупреждений о депрекации timm.models.layers
warnings.filterwarnings("ignore", message=".*timm.models.layers is deprecated.*")

def estimate_depth_midas(image, model_type="MiDaS_small"):
    """Оценка глубины с использованием предобученной модели MiDaS"""
    # Загрузка модели
    midas = torch.hub.load("intel-isl/MiDaS", model_type)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    midas.to(device)
    midas.eval()

    # Подготовка трансформации
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    transform = midas_transforms.small_transform if model_type == "MiDaS_small" else midas_transforms.dpt_transform

    # Подготовка изображения
    if isinstance(image, str):
        img = cv2.imread(image)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        img = image.copy()
        if len(img.shape) == 2:
            img = np.stack([img]*3, axis=2)
        if img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)

    # Преобразование изображения
    input_batch = transform(img).to(device)

    # Предсказание
    with torch.no_grad():
        prediction = midas(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    depth_map = prediction.cpu().numpy()

    # Нормализация
    depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min() + 1e-6)

    # Инвертирование карты глубины
    depth_map = 1.0 - depth_map

    return depth_map  # Инвертируем, чтобы ближние объекты были светлее