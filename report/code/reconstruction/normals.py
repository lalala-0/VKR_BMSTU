import cv2
import numpy as np
from scipy.ndimage import gaussian_filter, median_filter


def estimate_normals(shading, lighting_params, shadow_mask=None, multi_scale=True):
    """
    Оценка карты нормалей поверхности на основе карты затенения (shading)
    с улучшенным анализом теней и многомасштабным подходом.

    Parameters:
    -----------
    shading : numpy.ndarray
        Карта затенения (одноканальное изображение)
    lighting_params : dict
        Параметры освещения
    shadow_mask : numpy.ndarray, optional
        Маска теней (1 - свет, 0 - тень)
    multi_scale : bool, default=True
        Использовать многомасштабный подход для улучшения деталей

    Returns:
    --------
    tuple
        (normals, confidence_mask) - карта нормалей и маска уверенности
    """
    h, w = shading.shape

    # Преобразуем в float32 перед вычислением градиентов
    shading_float = shading.astype(np.float32)

    # Предварительная обработка - подавление шума при сохранении границ
    shading_denoised = cv2.bilateralFilter(shading_float, d=5, sigmaColor=0.1, sigmaSpace=2)

    # Обнаружение бликов для их последующего учета
    highlights_mask = shading_float > 0.9
    if np.any(highlights_mask):
        # Расширяем маску бликов
        kernel = np.ones((3, 3), np.uint8)
        highlights_mask = cv2.dilate(highlights_mask.astype(np.uint8), kernel, iterations=1)

    normals = np.zeros((h, w, 3), dtype=np.float32)
    final_confidence = np.ones((h, w), dtype=np.float32)

    # Параметры для многомасштабного анализа
    if multi_scale:
        scales = [3, 5, 9]
        scale_weights = [0.5, 0.3, 0.2]
    else:
        scales = [3]  # Только один масштаб, если multi_scale=False
        scale_weights = [1.0]

    # Многомасштабный анализ градиентов
    combined_grad_x = np.zeros((h, w), dtype=np.float32)
    combined_grad_y = np.zeros((h, w), dtype=np.float32)

    for scale_idx, ksize in enumerate(scales):
        # Оцениваем градиенты на разных масштабах
        grad_x_scale = cv2.Sobel(src=shading_denoised, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=ksize)
        grad_y_scale = cv2.Sobel(src=shading_denoised, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=ksize)

        # Вычисляем вес для текущего масштаба
        weight = scale_weights[scale_idx]

        # Накапливаем взвешенные градиенты
        combined_grad_x += grad_x_scale * weight
        combined_grad_y += grad_y_scale * weight

    # Сохраняем окончательные градиенты
    grad_x = combined_grad_x
    grad_y = combined_grad_y

    # Адаптивный коэффициент на основе статистики освещения
    brightness_std = lighting_params.get('brightness_std', 0.2)
    brightness_mean = lighting_params.get('brightness_mean', 0.5)

    # Более гибкая адаптация альфа в зависимости от характеристик освещения
    alpha_base = 0.2
    alpha = alpha_base * (1.0 / (brightness_std + 0.05))

    # Коррекция альфа в зависимости от средней яркости
    alpha *= np.clip(1.0 / (brightness_mean + 0.1), 0.5, 2.0)

    # Обработка теней, если маска предоставлена
    shadow_edges = None
    if shadow_mask is not None:
        # Нормализуем маску теней, если необходимо
        if shadow_mask.max() > 1.0:
            shadow_mask = shadow_mask / 255.0

        # Сглаживание маски теней для избежания резких переходов
        shadow_mask_smooth = gaussian_filter(shadow_mask, sigma=1.5)

        # Инвертируем маску (1 - нет тени, 0 - полная тень)
        shadow_influence = 1.0 - shadow_mask_smooth

        # Корректируем градиенты в затененных областях - более аккуратная коррекция с адаптивным коэффициентом
        shadow_grad_factor = np.clip(1.0 - shadow_influence * 0.65, 0.3, 1.0)
        grad_x = grad_x * shadow_grad_factor
        grad_y = grad_y * shadow_grad_factor

        # Улучшенное обнаружение границ теней с адаптивными порогами
        shadow_gray = (shadow_mask * 255).astype(np.uint8)
        mean_shadow = np.mean(shadow_gray)
        shadow_edges = cv2.Canny(shadow_gray,
                                max(10, int(mean_shadow*0.3)),
                                min(250, int(mean_shadow*1.5))) / 255.0

        # Улучшенная обработка границ теней
        shadow_edges = gaussian_filter(shadow_edges, sigma=1.2)

        # Создаем маску области вблизи границ теней для специальной обработки
        kernel = np.ones((5, 5), np.uint8)
        shadow_edges_region = cv2.dilate(shadow_edges.astype(np.uint8), kernel, iterations=2) / 255.0

    # Расчет нормалей с учетом информации о тенях
    normals[:, :, 0] = -alpha * grad_x
    normals[:, :, 1] = -alpha * grad_y
    normals[:, :, 2] = 1.0

    # Улучшенное сглаживание нормалей с учетом структуры изображения
    for c in range(3):
        # Адаптивная сила сглаживания в зависимости от компонента нормали
        sigma = 0.8 if c < 2 else 0.6  # Меньше сглаживаем z-компоненту для сохранения деталей

        # Анизотропная фильтрация для сглаживания с сохранением границ
        normals[:, :, c] = cv2.bilateralFilter(normals[:, :, c], d=5, sigmaColor=0.1, sigmaSpace=sigma)

    # Нормализация векторов нормалей
    norm = np.sqrt(np.sum(normals ** 2, axis=2, keepdims=True))
    normals = normals / (norm + 1e-10)

    # Улучшенная оценка достоверности нормалей
    gradient_magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
    max_magnitude = np.max(gradient_magnitude)
    if max_magnitude > 0:
        gradient_magnitude = gradient_magnitude / max_magnitude

    # Улучшенная коррекция весов на краях
    edge_strength = gradient_magnitude
    edge_correction = np.exp(-2.0 * edge_strength**2)  # Экспоненциальное убывание уверенности на сильных краях
    edge_correction = np.clip(edge_correction, 0.05, 1.0)

    # Базовая маска уверенности
    confidence_mask = (gradient_magnitude * 0.5 + 0.5) * edge_correction

    # Учет яркости с улучшенной функцией отклика
    brightness_response = np.exp(-4.0 * (shading - 0.5)**2)  # Гауссова функция с пиком при средней яркости
    brightness_weight = np.clip(brightness_response * 2.0, 0.2, 1.0)
    confidence_mask = confidence_mask * brightness_weight

    # Учитываем блики - снижаем уверенность в областях с бликами
    if np.any(highlights_mask):
        confidence_mask = confidence_mask * (1.0 - highlights_mask * 0.8)

    # Учитываем тени в маске уверенности
    if shadow_mask is not None:
        # Уменьшаем уверенность в затененных областях с плавным переходом
        shadow_confidence = np.clip(shadow_mask_smooth * 1.3, 0.1, 1.0)
        confidence_mask = confidence_mask * shadow_confidence

        # Дополнительно снижаем уверенность на границах теней и вблизи них
        if shadow_edges is not None:
            # Резко снижаем уверенность на самих границах
            confidence_mask = confidence_mask * (1.0 - shadow_edges * 0.7)
            # Плавно снижаем уверенность в области вблизи границ
            confidence_mask = confidence_mask * (1.0 - shadow_edges_region * 0.3)

    # Финальная обработка маски уверенности
    confidence_mask = gaussian_filter(confidence_mask, sigma=1.0)

    # Нормализация итоговой маски уверенности
    if np.max(confidence_mask) > 0:
        confidence_mask = confidence_mask / np.max(confidence_mask)

    return normals, confidence_mask