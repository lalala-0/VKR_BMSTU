import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

class SceneNetDataset(Dataset):
    def __init__(self, root_dir, protobuf_file, transform=None, max_samples=None):
        """
        Датасет SceneNetRGBD для оценки глубины

        Args:
            root_dir: Корневая директория с данными
            protobuf_file: Путь к protobuf-файлу с метаданными
            transform: Опциональные преобразования данных
            max_samples: Максимальное число образцов
        """
        # Здесь нужно загрузить protobuf-файл и сформировать список путей
        # к изображениям и соответствующим картам глубины
        self.root_dir = root_dir
        self.transform = transform

        # Загружаем пути из protobuf-файла
        import scenenet_pb2
        self.trajectories = []
        self.image_paths = []
        self.depth_paths = []

        with open(protobuf_file, 'rb') as f:
            trajectories = scenenet_pb2.Trajectories()
            trajectories.ParseFromString(f.read())

            for traj in trajectories.trajectories:
                render_path = os.path.join(root_dir, traj.render_path)
                for view in traj.views:
                    frame_num = view.frame_num
                    photo_path = os.path.join(render_path, 'photo', f'{frame_num}.jpg')
                    depth_path = os.path.join(render_path, 'depth', f'{frame_num}.png')

                    if os.path.exists(photo_path) and os.path.exists(depth_path):
                        self.image_paths.append(photo_path)
                        self.depth_paths.append(depth_path)

        if max_samples and max_samples < len(self.image_paths):
            indices = np.random.choice(len(self.image_paths), max_samples, replace=False)
            self.image_paths = [self.image_paths[i] for i in indices]
            self.depth_paths = [self.depth_paths[i] for i in indices]

        print(f"Загружено {len(self.image_paths)} изображений из SceneNetRGBD")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        depth_path = self.depth_paths[idx]

        image = Image.open(img_path).convert('RGB')

        # В SceneNetRGBD глубина хранится как uint16 в миллиметрах
        depth = np.array(Image.open(depth_path), dtype=np.float32) / 1000.0

        if self.transform:
            image = self.transform(image)

        # Преобразуем в тензоры
        image_tensor = torch.from_numpy(np.array(image).transpose(2, 0, 1)).float() / 255.0
        depth_tensor = torch.from_numpy(depth).float().unsqueeze(0)

        return {
            'image': image_tensor,
            'depth': depth_tensor,
            'image_path': img_path
        }