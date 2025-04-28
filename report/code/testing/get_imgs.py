import requests
from bs4 import BeautifulSoup
import os
import json


def scrape_shorpy_images(num_images=10, save_dir="shorpy_dataset"):
    """
    Скрапинг фотографий с Shorpy.com (только для образовательных целей)
    """
    os.makedirs(save_dir, exist_ok=True)
    base_url = "https://www.shorpy.com/recent-photos"

    metadata = []
    img_count = 0
    page = 1

    while img_count < num_images:
        r = requests.get(f"{base_url}?page={page}")
        soup = BeautifulSoup(r.content, 'html.parser')
        thumbnails = soup.find_all('div', class_='thumbnail')

        for thumb in thumbnails:
            if img_count >= num_images:
                break

            try:
                # Получение ссылки на детальную страницу
                detail_link = thumb.find('a')['href']
                detail_url = f"https://www.shorpy.com{detail_link}"

                # Получение деталей изображения
                detail_page = requests.get(detail_url)
                detail_soup = BeautifulSoup(detail_page.content, 'html.parser')

                # Получение полноразмерного изображения
                full_img = detail_soup.find('img', class_='shorpy-image')
                img_url = full_img['src']

                # Получение метаданных
                title = detail_soup.find('h1', class_='title').text.strip()
                date_text = detail_soup.find('div', class_='submitted').text
                # Извлечение года из текста даты (примерно)
                year = None
                for word in date_text.split():
                    if word.isdigit() and len(word) == 4:
                        year = int(word)
                        break

                # Скачивание изображения
                img_filename = f"shorpy_{img_count:03d}.jpg"
                img_path = os.path.join(save_dir, img_filename)
                with open(img_path, 'wb') as f:
                    img_data = requests.get(img_url).content
                    f.write(img_data)

                # Сохранение метаданных
                img_metadata = {
                    "image_id": f"shorpy_{img_count:03d}",
                    "filename": img_filename,
                    "source": "Shorpy Historical Archive",
                    "title": title,
                    "year": year,
                    "period": _determine_period(year),
                    "original_url": detail_url,
                    "category": _determine_category(title)
                }
                metadata.append(img_metadata)

                img_count += 1
                print(f"Скачано изображение {img_count}/{num_images}: {title}")

            except Exception as e:
                print(f"Ошибка при скачивании: {e}")

        page += 1

    # Сохранение метаданных в JSON
    with open(os.path.join(save_dir, "metadata.json"), 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    return metadata


def _determine_period(year):
    if not year:
        return "unknown"
    elif 1850 <= year <= 1900:
        return "early"
    elif 1901 <= year <= 1939:
        return "pre_war"
    elif 1940 <= year <= 1960:
        return "war_post_war"
    elif 1961 <= year <= 1990:
        return "late_analog"
    else:
        return "unknown"


def _determine_category(title):
    keywords = {
        "buildings": ["building", "architecture", "house", "hotel", "church", "temple"],
        "urban_panoramas": ["city", "street", "downtown", "skyline", "panorama"],
        "group_portraits": ["group", "family", "portrait", "class", "team"],
        "interiors": ["interior", "room", "inside", "office", "parlor"],
        "landscapes": ["landscape", "mountain", "sea", "river", "lake", "beach"]
    }

    title_lower = title.lower()
    for category, words in keywords.items():
        for word in words:
            if word in title_lower:
                return category

    return "uncategorized"