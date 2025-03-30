import cv2
import numpy as np


def visualize_results(results):
    preprocessed_vis = (results['preprocessed'] * 255).astype(np.uint8)
    if len(preprocessed_vis.shape) == 2:
        preprocessed_vis = cv2.cvtColor(preprocessed_vis, cv2.COLOR_GRAY2BGR)

    albedo_vis = (results['albedo'] * 255).astype(np.uint8)
    if len(albedo_vis.shape) == 2:
        albedo_vis = cv2.cvtColor(albedo_vis, cv2.COLOR_GRAY2BGR)

    shading_vis = (results['shading'] * 255).astype(np.uint8)
    if len(shading_vis.shape) == 2:
        shading_vis = cv2.cvtColor(shading_vis, cv2.COLOR_GRAY2BGR)

    # Визуализация нормалей
    normals = results['normals']
    normals_rgb = (normals + 1) * 0.5
    normals_vis = (normals_rgb * 255).astype(np.uint8)

    # Визуализация карты глубины
    depth_vis = np.uint8(results['final_depth'] * 255)
    depth_colored = cv2.applyColorMap(depth_vis, cv2.COLORMAP_INFERNO)

    # Визуализация сегментов
    segments_vis = np.zeros_like(preprocessed_vis)
    segments = results['segments_mask']

    # Отрисовка прямоугольных сегментов
    for segment_id, bbox in results['bboxes'].items():
        y1, y2, x1, x2 = bbox

        # Генерируем цвет в зависимости от segment_id
        color = np.array([
            (segment_id * 67) % 256,
            (segment_id * 111) % 256,
            (segment_id * 189) % 256
        ])

        # Рисуем прямоугольник
        rect_img = segments_vis.copy()
        cv2.rectangle(rect_img, (x1, y1), (x2, y2), color.tolist(), 2)

        # Маска сегмента
        segment_mask = (segments[y1:y2, x1:x2] == segment_id).astype(np.uint8)

        # Применяем цвет к точкам внутри сегмента
        for c in range(3):
            segments_vis[y1:y2, x1:x2, c] = np.where(
                segment_mask > 0,
                color[c] * 0.7,
                segments_vis[y1:y2, x1:x2, c]
            )

        # Добавляем прямоугольник
        segments_vis = cv2.addWeighted(segments_vis, 0.7, rect_img, 0.3, 0)

    # Визуализация рельефа
    depth_relief = results['final_depth'].copy()
    depth_relief_uint8 = (depth_relief * 255).astype(np.uint8)

    dx = cv2.Sobel(depth_relief_uint8, cv2.CV_8U, 1, 0, ksize=3)
    dy = cv2.Sobel(depth_relief_uint8, cv2.CV_8U, 0, 1, ksize=3)

    relief_factor = 3.0
    dx = dx.astype(np.float32) / 255.0 * relief_factor
    dy = dy.astype(np.float32) / 255.0 * relief_factor

    relief = np.clip(depth_relief + dx * 0.5 - dy * 0.5, 0, 1)
    relief_vis = (relief * 255).astype(np.uint8)
    relief_colored = cv2.applyColorMap(relief_vis, cv2.COLORMAP_BONE)

    visualization = {
        'preprocessed': preprocessed_vis,
        'albedo': albedo_vis,
        'shading': shading_vis,
        'normals': normals_vis,
        'depth': depth_vis,
        'depth_colored': depth_colored,
        'segments': segments_vis,
        'relief_colored': relief_colored
    }

    return visualization
