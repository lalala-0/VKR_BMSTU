import os

import cv2
import h5py
import numpy as np
import torch
from PIL.Image import Image
from torch.utils.data import Dataset


class DepthTestDataset(Dataset):
    def __init__(self, root_dir, transform=None, aging_effects=None):
        """
        Args:
            root_dir (str): Директория с изображениями и картами глубины
            transform: Преобразования для изображений
            aging_effects: Функции для имитации старения фотографий
        """
        self.root_dir = root_dir
        self.transform = transform
        self.aging_effects = aging_effects
        self.samples = self._load_samples()

    def _load_samples(self):
        """
        Загружает пары RGB-глубина из каталогов rgb и depth
        """
        samples = []
        rgb_dir = os.path.join(self.root_dir, 'rgb')
        depth_dir = os.path.join(self.root_dir, 'depth')

        if not os.path.exists(rgb_dir):
            raise ValueError(f"Директория RGB изображений не найдена: {rgb_dir}")
        if not os.path.exists(depth_dir):
            raise ValueError(f"Директория карт глубины не найдена: {depth_dir}")

        # Получаем список файлов RGB
        rgb_files = sorted([f for f in os.listdir(rgb_dir)
                            if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

        # Для каждого RGB файла ищем соответствующую карту глубины
        for rgb_file in rgb_files:
            base_name = os.path.splitext(rgb_file)[0]

            # Ищем соответствующую карту глубины в разных форматах
            depth_found = False
            for ext in ['.png', '.jpg', '.jpeg', '.npy', '.exr']:
                depth_path = os.path.join(depth_dir, base_name + ext)
                if os.path.exists(depth_path):
                    samples.append({
                        'rgb_path': os.path.join(rgb_dir, rgb_file),
                        'depth_path': depth_path
                    })
                    depth_found = True
                    break

            if not depth_found:
                print(f"Предупреждение: не найдена карта глубины для {rgb_file}")

        print(f"Загружено {len(samples)} пар изображение-глубина из {self.root_dir}")
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # Загрузка RGB-изображения и карты глубины
        rgb_image = load_rgb(self.samples[idx]['rgb_path'])
        depth_map = load_depth(self.samples[idx]['depth_path'])

        # Применение эффектов старения
        if self.aging_effects:
            rgb_image = self.aging_effects(rgb_image)

        # Применение трансформаций
        if self.transform:
            rgb_image = self.transform(rgb_image)
        else:
            # Преобразуем в тензор PyTorch
            rgb_image = torch.from_numpy(rgb_image.transpose((2, 0, 1))).float() / 255.0

        # Преобразуем карту глубины в тензор
        depth_tensor = torch.from_numpy(depth_map).float()

        return {'image': rgb_image, 'depth': depth_tensor}


class NYUDepthDataset(Dataset):
    """
    Класс для работы с NYU Depth Dataset V2
    """

    def __init__(self, dataset_path, transform=None, aging_effects=None, split='test', max_samples=None):
        """
        Args:
            dataset_path: Путь к файлу .mat с датасетом NYU Depth V2
            transform: Преобразования изображений
            aging_effects: Эффекты старения для изображений
            split: Разделение данных ('train' или 'test')
            max_samples: Максимальное количество используемых образцов
        """
        self.dataset_path = dataset_path
        self.transform = transform
        self.aging_effects = aging_effects
        self.split = split

        # Загружаем датасет
        print(f"Загрузка NYU Depth Dataset V2 из {dataset_path}...")
        self.data = h5py.File(dataset_path, 'r')

        # Получаем изображения и карты глубины
        self.images = np.array(self.data['images'])
        self.depths = np.array(self.data['depths'])

        # Необходимо транспонировать данные из-за формата HDF5
        self.images = np.transpose(self.images, (3, 2, 1, 0))
        self.depths = np.transpose(self.depths, (2, 1, 0))

        # Делим на тренировочную и тестовую части
        total_samples = self.images.shape[3]
        if split == 'test':
            # По соглашению, используем последние 20% для тестирования
            start_idx = int(total_samples * 0.8)
            self.images = self.images[:, :, :, start_idx:]
            self.depths = self.depths[:, :, start_idx:]
        else:
            # Тренировочная часть - первые 80%
            end_idx = int(total_samples * 0.8)
            self.images = self.images[:, :, :, :end_idx]
            self.depths = self.depths[:, :, :end_idx]

        # Ограничиваем количество образцов, если указано
        if max_samples is not None and max_samples < self.images.shape[3]:
            self.images = self.images[:, :, :, :max_samples]
            self.depths = self.depths[:, :, :max_samples]

        print(f"Загружено {self.images.shape[3]} образцов из NYU Depth Dataset V2")

    def __len__(self):
        return self.images.shape[3]

    def __getitem__(self, idx):
        # Получаем RGB-изображение и карту глубины
        rgb_image = self.images[:, :, :, idx]
        depth_map = self.depths[:, :, idx]

        # Приводим к стандартному формату
        rgb_image = np.transpose(rgb_image, (1, 2, 0))

        # Применяем эффекты старения, если нужно
        if self.aging_effects:
            rgb_image = self.aging_effects(rgb_image)

        # Применяем трансформации, если есть
        if self.transform:
            rgb_image = self.transform(rgb_image)
        else:
            # Преобразуем в тензор PyTorch
            rgb_image = torch.from_numpy(rgb_image.transpose((2, 0, 1))).float() / 255.0

        # Преобразуем карту глубины в тензор
        depth_map = torch.from_numpy(depth_map).float()

        return {'image': rgb_image, 'depth': depth_map}


class KittiDepthDataset(Dataset):
    """
    Класс для работы с KITTI Depth Dataset

    Структура датасета:
    root/
        training/
            image_2/            # RGB изображения
                000000_10.png
                ...
            depth_map/          # Карты глубины
                000000_10.png
                ...
    """

    def __init__(self, root_dir, transform=None, aging_effects=None, split='train', max_samples=None):
        self.root_dir = root_dir
        self.transform = transform
        self.aging_effects = aging_effects
        self.split = split

        # Определяем пути к подкаталогам
        self.images_dir = os.path.join(root_dir, split, "image_2")
        self.depth_dir = os.path.join(root_dir, split, "depth_map")

        # Проверяем существование директорий
        if not os.path.exists(self.images_dir):
            raise ValueError(f"Директория изображений не найдена: {self.images_dir}")
        if not os.path.exists(self.depth_dir) and split == 'train':
            raise ValueError(f"Директория карт глубины не найдена: {self.depth_dir}")

        # Получаем список файлов
        self.image_files = sorted([f for f in os.listdir(self.images_dir)
                                   if f.endswith(('.png', '.jpg'))])

        # Ограничиваем количество образцов, если указано
        if max_samples and max_samples < len(self.image_files):
            self.image_files = self.image_files[:max_samples]

        print(f"Загружено {len(self.image_files)} образцов из KITTI Dataset")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_name = self.image_files[idx]
        img_path = os.path.join(self.images_dir, image_name)

        # Загружаем RGB изображение
        rgb_image = load_rgb(img_path)

        # Загружаем карту глубины, если доступна
        if self.split == 'train':
            depth_path = os.path.join(self.depth_dir, image_name)
            depth_map = load_depth(depth_path)
            depth_tensor = torch.from_numpy(depth_map).float()
        else:
            # Для тестового набора карты глубины может не быть
            depth_tensor = torch.zeros((rgb_image.shape[0], rgb_image.shape[1]))

        # Применяем эффекты старения, если нужно
        if self.aging_effects:
            rgb_image = self.aging_effects(rgb_image)

        # Применяем трансформации
        if self.transform:
            rgb_image = self.transform(rgb_image)
        else:
            # Преобразуем в тензор PyTorch
            rgb_image = torch.from_numpy(rgb_image.transpose((2, 0, 1))).float() / 255.0

        return {'image': rgb_image, 'depth': depth_tensor}




def load_rgb(path):
    """
    Загружает RGB-изображение с диска

    Args:
        path (str): Путь к изображению

    Returns:
        np.ndarray: RGB-изображение в формате NumPy
    """
    try:
        # Загрузка с помощью OpenCV и конвертация в RGB
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(f"Не удалось загрузить изображение {path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img
    except Exception as e:
        # Альтернативный метод с PIL
        img = np.array(Image.open(path).convert('RGB'))
        return img

def load_depth(path):
    """
    Загружает карту глубины с диска в зависимости от формата

    Args:
        path (str): Путь к карте глубины

    Returns:
        np.ndarray: Карта глубины в формате NumPy
    """
    extension = os.path.splitext(path)[1].lower()

    if extension == '.png' or extension == '.jpg' or extension == '.jpeg':
        # Карты глубины в формате изображений (например, NYU Depth)
        depth = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if len(depth.shape) > 2:
            depth = depth[:, :, 0]  # Использовать только первый канал
        return depth.astype(np.float32) / 1000.0  # Нормализация (обычно в мм)

    elif extension == '.npy':
        # Карты глубины в формате NumPy
        return np.load(path).astype(np.float32)

    elif extension == '.exr':
        # Карты глубины в формате OpenEXR (часто используется в синтетических датасетах)
        try:
            import OpenEXR
            import Imath

            exr_file = OpenEXR.InputFile(path)
            header = exr_file.header()
            dw = header['dataWindow']
            size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)

            pt = Imath.PixelType(Imath.PixelType.FLOAT)
            depth_str = exr_file.channel('Z', pt)

            depth = np.frombuffer(depth_str, dtype=np.float32)
            depth.shape = (size[1], size[0])
            return depth
        except ImportError:
            raise ImportError("Для работы с EXR требуется установить библиотеки OpenEXR и Imath")

    else:
        raise ValueError(f"Неподдерживаемый формат карты глубины: {extension}")