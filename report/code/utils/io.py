import os
import cv2

def save_results(visualization, output_dir="./out"):
    """
    Сохраняет результаты визуализации в указанную директорию
    """
    # Создаем директорию, если она не существует
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Сохраняем результаты визуализации
    cv2.imwrite(os.path.join(output_dir, "preprocessed.png"), visualization['preprocessed'])
    cv2.imwrite(os.path.join(output_dir, "albedo.png"), visualization['albedo'])
    cv2.imwrite(os.path.join(output_dir, "shading.png"), visualization['shading'])
    cv2.imwrite(os.path.join(output_dir, "normals.png"), visualization['normals'])
    cv2.imwrite(os.path.join(output_dir, "depth_map.png"), visualization['depth'])
    cv2.imwrite(os.path.join(output_dir, "depth_colored.png"), visualization['depth_colored'])
    cv2.imwrite(os.path.join(output_dir, "segments.png"), visualization['segments'])
    cv2.imwrite(os.path.join(output_dir, "relief_colored.png"), visualization['relief_colored'])

    print(f"Результаты сохранены в директорию {output_dir}")

def check_and_create_dirs(directories):
    """
    Проверяет и создает указанные директории, если они не существуют
    """
    for directory, description in directories.items():
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Создана директория {description}: {directory}")