import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Загружаем модель MiDaS
model_type = "DPT_Large"  # Можно попробовать "MiDaS_small" для скорости
model = torch.hub.load("intel-isl/MiDaS", model_type)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# Загружаем трансформации для DPT_Large
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform = midas_transforms.dpt_transform  # Для DPT

def estimate_depth(image_path):
    # Загружаем изображение (PIL)
    img = Image.open(image_path).convert("RGB")
    img = np.array(img, dtype=np.float32)  # Преобразование в numpy

    # Применяем трансформации
    img_tensor = transform(img).to(device)
    print("Форма тензора после трансформации:", img_tensor.shape)

    # Прогоняем через модель
    with torch.no_grad():
        depth_map = model(img_tensor)

    print("Форма карты глубины до squeeze:", depth_map.shape)

    # Преобразование карты глубины
    depth_map = depth_map.squeeze().cpu().numpy()
    if len(depth_map.shape) == 3:  # Если [1, 1, H, W]
        depth_map = depth_map[0]

    # Нормализация для отображения
    depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())
    
    return np.array(img), depth_map

def show_result(image, depth_map):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(image)
    axes[0].set_title("Исходное изображение")
    axes[0].axis("off")
    
    axes[1].imshow(depth_map, cmap="inferno")
    axes[1].set_title("Оценка глубины")
    axes[1].axis("off")
    
    plt.show()

# Запуск на тестовом изображении
image_path = "old_photo.jpg"  # Укажи путь к фото
image, depth = estimate_depth(image_path)
show_result(image, depth)
