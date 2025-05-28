import glob
import scipy.io as sio
import h5py
import os
import cv2
import time
from abc import ABC, abstractmethod
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

# Импорты из других модулей проекта
from testing.metrics import compute_depth_metrics
from testing.aging import apply_aging_effects
from testing.inout import create_result_directories, print_metrics_summary, save_metrics_to_json, visualize_depth_results
from reconstruction.depth_reconstruction import ArchivalDepthReconstruction
from testing.scenenet_dataset import SceneNetDataset
from testing.scenenet_dataset import SceneNetDataset as SceneNetLoader


def initialize_depth_reconstructor(args):
    """
    Создает и инициализирует модель для восстановления глубины

    Args:
        args: Аргументы командной строки

    Returns:
        Экземпляр ArchivalDepthReconstruction
    """
    reconstructor = ArchivalDepthReconstruction(
        iterations=args.iterations if hasattr(args, 'iterations') else 200,
        lambda_smooth=0.2,
        brightness_boost=args.brightness if hasattr(args, 'brightness') else 1.8,
        contrast_boost=args.contrast if hasattr(args, 'contrast') else 1.5,
        n_color_segments=args.segments if hasattr(args, 'segments') else 8,
        method=args.method,
        midas_model=args.midas_model if hasattr(args, 'midas_model') else 'MiDaS_small'
    )
    return reconstructor


class DepthDataset(ABC):
    """Абстрактный класс для работы с датасетами глубины"""

    def __init__(self, args):
        self.args = args
        self.dataset_name = None
        self.sample_count = 0

    @abstractmethod
    def load_data(self):
        """Загрузка данных из датасета"""
        pass

    @abstractmethod
    def process_sample(self, sample_idx, reconstructor):
        """Обработка одного образца из датасета"""
        pass

    def get_name(self):
        """Получение имени датасета"""
        return self.dataset_name

    def get_sample_count(self):
        """Получение количества образцов в датасете"""
        return self.sample_count


class Make3DDataset(DepthDataset):
    """Класс для работы с датасетом Make3D"""

    def __init__(self, args):
        super().__init__(args)
        self.dataset_name = 'make3d'
        self.image_files = []
        self.depth_files = []

    def load_data(self):
        dataset_path = self.args.test_dataset
        images_path = os.path.join(dataset_path, 'Images')
        depth_path = os.path.join(dataset_path, 'Depths')

        # Получаем списки файлов
        self.image_files = sorted(glob.glob(os.path.join(images_path, '*.jpg')))
        self.depth_files = sorted(glob.glob(os.path.join(depth_path, '*.mat')))

        if not self.image_files:
            print(f"Ошибка: не найдены изображения в {images_path}")
            return False

        if not self.depth_files:
            print(f"Ошибка: не найдены файлы глубины в {depth_path}")
            return False

        # Ограничиваем количество образцов
        max_samples = self.args.max_samples if hasattr(self.args, 'max_samples') and self.args.max_samples else len(self.image_files)
        self.sample_count = min(max_samples, len(self.image_files))

        print(f"Найдено {len(self.image_files)} изображений в датасете Make3D")
        print(f"Будет обработано {self.sample_count} изображений")
        return True

    def process_sample(self, sample_idx, reconstructor):
        image_file = self.image_files[sample_idx]
        depth_file = self.depth_files[sample_idx]

        # Загружаем изображение
        image = cv2.imread(image_file)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Загружаем карту глубины из .mat файла
        depth_mat = sio.loadmat(depth_file)

        # В датасете Make3D глубина часто хранится под ключом 'depthLayers' или 'Position3DGrid'
        if 'depthLayers' in depth_mat:
            gt_depth = depth_mat['depthLayers']
        elif 'Position3DGrid' in depth_mat:
            gt_depth = depth_mat['Position3DGrid'][:, :, 3]
        else:
            # Попробуем найти подходящий ключ (исключая служебные ключи)
            depth_keys = [k for k in depth_mat.keys() if not k.startswith('__')]
            if depth_keys:
                gt_depth = depth_mat[depth_keys[0]]
                print(f"Используется ключ {depth_keys[0]} для данных глубины")
            else:
                raise KeyError(f"Не найден ключ с данными глубины в файле {depth_file}")

        # Применяем эффекты старения, если требуется
        image = apply_aging_effects(image, self.args)

        # Обрабатываем изображение моделью
        results = reconstructor.process_numpy(image)
        pred_depth = results['depth_map']

        # Приводим к одинаковым размерам
        if pred_depth.shape != gt_depth.shape:
            pred_depth = cv2.resize(pred_depth, (gt_depth.shape[1], gt_depth.shape[0]))

        # Маска валидных пикселей (где глубина > 0)
        valid_mask = (gt_depth > 0)

        # Масштабирование предсказанной глубины к диапазону GT
        if valid_mask.sum() > 0:
            scale = np.median(gt_depth[valid_mask]) / np.median(pred_depth[valid_mask])
            pred_depth = pred_depth * scale

        # Вычисляем метрики
        img_metrics = compute_depth_metrics(pred_depth, gt_depth, valid_mask)


        # Формируем данные для сохранения
        img_data = {k: float(v) for k, v in img_metrics.items()}
        img_data['image_id'] = sample_idx
        img_data['image_path'] = image_file

        return {
            'image': image,
            'gt_depth': gt_depth,
            'pred_depth': pred_depth,
            'metrics': img_data
        }


class SceneNetDataset(DepthDataset):
    """Класс для работы с датасетом SceneNetRGBD"""

    def __init__(self, args):
        super().__init__(args)
        self.dataset_name = 'scenenet'
        self.dataset = None
        self.dataloader = None

    def load_data(self):
        # Определяем директории protobuf и корневую директорию датасета
        protobuf_path = os.path.join(self.args.test_dataset, 'scenenet_rgbd_val.pb')
        dataset_root = self.args.test_dataset

        # Применение эффектов старения к изображениям, если требуется
        transform = None

        # Создаем датасет и загрузчик данных
        self.dataset = SceneNetLoader(
            root_dir=dataset_root,
            protobuf_file=protobuf_path,
            transform=transform,
            max_samples=self.args.max_samples
        )

        self.dataloader = DataLoader(
            self.dataset,
            batch_size=self.args.batch_size,
            num_workers=self.args.workers,
            shuffle=False
        )

        self.sample_count = len(self.dataset)
        print(f"Начинаем тестирование на {self.sample_count} изображениях...")
        return True

    def process_batch(self, reconstructor):
        all_results = []
        batch_idx = 0

        for batch in tqdm(self.dataloader):
            images = batch['image']
            gt_depths = batch['depth_map']
            paths = batch['image_path']

            for i in range(len(images)):
                image = images[i].numpy().transpose(1, 2, 0) * 255
                image = image.astype(np.uint8)

                # Применение эффектов старения
                image = apply_aging_effects(image, self.args)

                # Обработка изображения моделью
                results = reconstructor.process_numpy(image)
                pred_depth = results['depth_map']

                # Вычисление метрик
                gt_depth = gt_depths[i].squeeze().numpy()
                valid_mask = (gt_depth > 0)

                # Масштабирование предсказанной глубины к диапазону GT
                if valid_mask.sum() > 0:
                    pred_depth = pred_depth * (np.median(gt_depth[valid_mask]) / np.median(pred_depth[valid_mask]))

                # Вычисляем метрики
                img_metrics = compute_depth_metrics(pred_depth, gt_depth, valid_mask)

                # Сохраняем метрики
                img_data = {k: float(v) for k, v in img_metrics.items()}
                img_data['image_id'] = batch_idx
                img_data['image_path'] = paths[i]

                all_results.append({
                    'image': image,
                    'gt_depth': gt_depth,
                    'pred_depth': pred_depth,
                    'metrics': img_data
                })

                batch_idx += 1

        return all_results

    def process_sample(self, sample_idx, reconstructor):
        # Метод не используется для SceneNetDataset, так как обработка идет пакетами
        pass


class NYUDataset(DepthDataset):
    """Класс для работы с датасетом NYU Depth"""

    def __init__(self, args):
        super().__init__(args)
        self.dataset_name = 'nyu'
        self.images = None
        self.depths = None
        self.results_dir = None

    def load_data(self):
        print(f"Загрузка NYU Depth Dataset V2 из {self.args.test_dataset}...")
        data = h5py.File(self.args.test_dataset, 'r')

        # Получаем массивы изображений и глубины
        self.images = np.array(data['images'])
        self.depths = np.array(data['depths'])
        data.close()  # Закрываем файл h5py сразу после загрузки

        # Ограничиваем количество образцов
        max_samples = self.args.max_samples if hasattr(self.args, 'max_samples') and self.args.max_samples else len(self.images)
        self.sample_count = min(max_samples, len(self.images))

        print(f"Загружено {len(self.images)} образцов из NYU Depth Dataset V2")
        print(f"Будет обработано {self.sample_count} образцов")
        return True

    def process_sample(self, sample_idx, reconstructor):
        # Преобразуем формат изображения
        image = np.transpose(self.images[sample_idx], (1, 2, 0))  # [C,H,W] -> [H,W,C]
        gt_depth = self.depths[sample_idx]

        # Применяем эффекты старения
        image = apply_aging_effects(image, self.args)

        # Обрабатываем изображение
        results = reconstructor.process_numpy(image)

        pred_depth = results.get('final_depth', results.get('depth_map'))

        # Приводим к одинаковым размерам
        if pred_depth.shape != gt_depth.shape:
            pred_depth = cv2.resize(pred_depth, (gt_depth.shape[1], gt_depth.shape[0]))

        # Вычисляем метрики для этого изображения
        img_metrics = compute_depth_metrics(
            pred_depth[np.newaxis, ...],
            gt_depth[np.newaxis, ...]
        )

        # Сохраняем метрики
        img_data = {k: float(v) for k, v in img_metrics.items()}
        img_data['image_id'] = sample_idx

        return {
            'image': image,
            'gt_depth': gt_depth,
            'pred_depth': pred_depth,
            'metrics': img_data
        }

    def set_results_dir(self, results_dir):
        self.results_dir = results_dir


class LightFieldDataset(DepthDataset):
    """Класс для работы с датасетом 4D Light Field"""

    def __init__(self, args):
        super().__init__(args)
        self.dataset_name = 'lightfield'
        self.image_files = []
        self.depth_files = []
        self.scene_dirs = []
        self.dataset_path = "./testing/datasets/lightfield"  # Путь к датасету по умолчанию

    def load_data(self):
        dataset_path = self.dataset_path

        # Ищем директории сцен
        self.scene_dirs = [d for d in os.listdir(dataset_path)
                           if os.path.isdir(os.path.join(dataset_path, d))]

        if not self.scene_dirs:
            print(f"Ошибка: не найдены сцены в {dataset_path}")
            return False

        print(f"Найдено {len(self.scene_dirs)} сцен в датасете Light Field")

        # Для каждой сцены находим центральное изображение и карту глубины
        for scene in self.scene_dirs:
            scene_dir = os.path.join(dataset_path, scene)

            # Центральное изображение
            img_path = os.path.join(scene_dir, 'input_Cam005.png')

            # Карта глубины
            depth_path = os.path.join(scene_dir, 'gt_disp_lowres.pfm')

            if os.path.exists(img_path) and os.path.exists(depth_path):
                self.image_files.append(img_path)
                self.depth_files.append(depth_path)

        # Ограничиваем количество образцов
        max_samples = self.args.max_samples if hasattr(self.args, 'max_samples') and self.args.max_samples else len(
            self.image_files)
        self.sample_count = min(max_samples, len(self.image_files))

        print(f"Будет обработано {self.sample_count} изображений")
        return True

    def _read_pfm(self, file_path):
        """Чтение PFM файла с картой глубины"""
        with open(file_path, 'rb') as file:
            header = file.readline().decode('utf-8').strip()
            if header not in ['PF', 'Pf']:  # Поддержка обоих вариантов заголовка
                raise ValueError(f'Неподдерживаемый формат PFM: {header}')

            dim_line = file.readline().decode('utf-8').strip()
            width, height = map(int, dim_line.split())

            scale_line = file.readline().decode('utf-8').strip()
            scale = float(scale_line)
            endian = '<' if scale < 0 else '>'

            # Для 'Pf' используется одноканальный формат (только глубина)
            channels = 1 if header == 'Pf' else 3

            data = np.fromfile(file, endian + 'f')
            shape = (height, width) if channels == 1 else (height, width, channels)
            data = data.reshape(shape)

            # Если scale < 0, изображение перевернуто
            if scale < 0:
                data = np.flipud(data)

            return data

    def process_sample(self, sample_idx, reconstructor):
        # Загрузка изображения
        img_path = self.image_files[sample_idx]
        depth_path = self.depth_files[sample_idx]

        # Загружаем изображение
        image = cv2.imread(img_path)
        if image is None:
            print(f"Ошибка: не удалось загрузить изображение {img_path}")
            return None

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Загружаем карту глубины в формате PFM
        try:
            gt_depth = self._read_pfm(depth_path)
        except Exception as e:
            print(f"Ошибка при чтении карты глубины {depth_path}: {str(e)}")
            return None

        # Применяем эффекты старения
        image = apply_aging_effects(image, self.args)

        # Обрабатываем изображение моделью
        results = reconstructor.process_numpy(image)
        pred_depth = results['depth_map']

        # Приводим к одинаковым размерам
        if pred_depth.shape != gt_depth.shape:
            pred_depth = cv2.resize(pred_depth, (gt_depth.shape[1], gt_depth.shape[0]))

        # Маска валидных пикселей (где глубина > 0)
        valid_mask = (gt_depth > 0)

        # Масштабирование предсказанной глубины к диапазону GT
        if valid_mask.sum() > 0:
            scale = np.median(gt_depth[valid_mask]) / np.median(pred_depth[valid_mask])
            pred_depth = pred_depth * scale

        # Вычисляем метрики
        img_metrics = compute_depth_metrics(pred_depth, gt_depth, valid_mask)

        # Формируем данные
        img_data = {k: float(v) for k, v in img_metrics.items()}
        img_data['image_id'] = sample_idx
        img_data['image_path'] = img_path

        return {
            'image': image,
            'gt_depth': gt_depth,
            'pred_depth': pred_depth,
            'metrics': img_data
        }

class DatasetFactory:
    """Фабрика для создания экземпляров датасетов"""

    @staticmethod
    def get_dataset(dataset_type, args):
        if dataset_type.lower() == 'make3d':
            return Make3DDataset(args)
        elif dataset_type.lower() == 'scenenet':
            return SceneNetDataset(args)
        elif dataset_type.lower() == 'nyu':
            return NYUDataset(args)
        elif dataset_type.lower() == 'lightfield':
            return LightFieldDataset(args)
        else:
            raise ValueError(f"Неизвестный тип датасета: {dataset_type}")

def run_testing(args):
    """
    Общая функция для запуска тестирования на различных датасетах

    Args:
        args: Аргументы командной строки с обязательным параметром dataset_type

    Returns:
        dict: Усредненные метрики
    """
    # Проверка наличия аргумента dataset_type
    if not hasattr(args, 'dataset_type'):
        raise ValueError("Не указан тип датасета (dataset_type)")

    # Создаем датасет через фабрику
    dataset = DatasetFactory.get_dataset(args.dataset_type, args)

    # Загружаем данные
    if not dataset.load_data():
        print(f"Ошибка загрузки данных из датасета {args.dataset_type}")
        return None

    # Создаем директории для результатов
    results_dir, vis_dir = create_result_directories(dataset.get_name(), args.method)

    # Для NYU датасета устанавливаем директорию для временных файлов
    if args.dataset_type.lower() == 'nyu':
        dataset.set_results_dir(results_dir)

    # Создаем модель реконструкции
    reconstructor = initialize_depth_reconstructor(args)

    # Метрики
    all_metrics = []
    metrics_sum = {
        'mae': 0, 'rmse': 0, 'rel': 0, 'delta1': 0, 'delta2': 0, 'delta3': 0,
    }

    # Тестирование
    print(f"Запуск тестирования метода {args.method} на датасете {args.dataset_type}...")
    start_time = time.time()

    # Для SceneNet обрабатываем все сразу батчами
    if args.dataset_type.lower() == 'scenenet':
        all_results = dataset.process_batch(reconstructor)
        for result in all_results:
            all_metrics.append(result['metrics'])

            # Накапливаем суммарные метрики
            for k, v in result['metrics'].items():
                if k in metrics_sum:
                    metrics_sum[k] += v

            # Визуализация (каждые 10 изображений)
            if result['metrics']['image_id'] % 10 == 0:
                visualize_depth_results(
                    result['image'], result['gt_depth'], result['pred_depth'],
                    os.path.join(vis_dir, f"sample_{result['metrics']['image_id']}.png")
                )
    else:
        # Для остальных датасетов обрабатываем по одному образцу
        sample_count = dataset.get_sample_count()

        for i in tqdm(range(sample_count)):
            try:
                result = dataset.process_sample(i, reconstructor)

                all_metrics.append(result['metrics'])

                # Накапливаем суммарные метрики
                for k, v in result['metrics'].items():
                    if k in metrics_sum:
                        metrics_sum[k] += v

                # Визуализация (каждые 10 изображений или последнее)
                if i % 10 == 0 or i == sample_count - 1:
                    visualize_depth_results(
                        result['image'], result['gt_depth'], result['pred_depth'],
                        os.path.join(vis_dir, f"sample_{i}.png")
                    )

                if 'mae' in result['metrics']:
                    print(f"Обработано изображение {i + 1}/{sample_count}, MAE: {result['metrics']['mae']:.4f}")

            except Exception as e:
                print(f"Ошибка при обработке изображения {i}: {str(e)}")

    # Вычисляем средние метрики
    sample_count = dataset.get_sample_count()
    metrics_avg = {k: v / sample_count for k, v in metrics_sum.items() if v > 0}

    # Выводим результаты
    print_metrics_summary(metrics_avg, args.dataset_type)

    # Сохраняем результаты в файл
    results_file = save_metrics_to_json(
        metrics_avg, all_metrics, results_dir, dataset.get_name(),
        args.method, sample_count, time.time() - start_time
    )

    print(f"Результаты сохранены в {results_file}")
    return metrics_avg


# Для совместимости со старым кодом - обертки для конкретных датасетов
def run_testing_make3d(args):
    args.dataset_type = 'make3d'
    return run_testing(args)


def run_testing_scenenet(args):
    args.dataset_type = 'scenenet'
    return run_testing(args)


def run_testing_nyu(args):
    args.dataset_type = 'nyu'
    return run_testing(args)


def run_testing_lightfield(args):
    args.dataset_type = 'lightfield'
    return run_testing(args)