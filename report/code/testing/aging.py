import numpy as np
import cv2

class AgingEffects:
    @staticmethod
    def add_noise(image, intensity=0.1):
        """Добавляет шум, имитирующий зернистость старых фото"""
        noise = np.random.normal(0, intensity * 255, image.shape).astype(np.uint8)
        return np.clip(image + noise, 0, 255).astype(np.uint8)

    @staticmethod
    def add_scratches(image, num_scratches=20):
        """Добавляет царапины на изображение"""
        result = image.copy()
        h, w = image.shape[:2]

        for _ in range(num_scratches):
            x1, y1 = np.random.randint(0, w), np.random.randint(0, h)
            x2, y2 = np.random.randint(0, w), np.random.randint(0, h)
            thickness = np.random.randint(1, 3)
            color = (255, 255, 255) if np.random.random() > 0.5 else (0, 0, 0)
            cv2.line(result, (x1, y1), (x2, y2), color, thickness)

        return result

    @staticmethod
    def sepia_effect(image):
        """Применяет эффект сепии"""
        sepia_filter = np.array([[0.393, 0.769, 0.189],
                                [0.349, 0.686, 0.168],
                                [0.272, 0.534, 0.131]])
        sepia_image = cv2.transform(image, sepia_filter)
        return np.clip(sepia_image, 0, 255).astype(np.uint8)

    @staticmethod
    def vignette(image, intensity=0.5):
        """Добавляет виньетирование по краям изображения"""
        h, w = image.shape[:2]
        kernel_x = cv2.getGaussianKernel(w, w/3)
        kernel_y = cv2.getGaussianKernel(h, h/3)
        kernel = kernel_y * kernel_x.T

        mask = 255 * kernel / np.linalg.norm(kernel)
        mask = (mask * (1 - intensity) + intensity * 255).astype(np.uint8)

        result = image.copy()
        for i in range(3):
            result[:, :, i] = result[:, :, i] * mask / 255

        return result