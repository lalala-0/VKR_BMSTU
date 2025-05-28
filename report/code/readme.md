# Использование
Запуск с различными методами реконструкции

## Метод сегментации (по умолчанию)

`python main.py --image_path "E:\\VKR_BMSTU\\report\\code\\img\\testimg5.png" --method segmentation`

Параметры:
- segments или -s: количество цветовых сегментов (по умолчанию: 8)
- iterations или -i: количество итераций для восстановления глубины (по умолчанию: 200)

## Метод MiDaS

`python main.py --image_path "D:\\VKR_BMSTU\\report\\code\\img\\testimg5.png" --method midas`

Параметры:
- midas-model: модель MiDaS для использования (по умолчанию: MiDaS_small)
    - MiDaS_small: быстрая работа, но менее точная
    - DPT_Hybrid: баланс между скоростью и качеством
    - DPT_Large: лучшее качество, но более медленная работа

## Гибридный метод (MiDaS + сегментация)

`python main.py --image_path "D:\\VKR_BMSTU\\report\\code\\img\\img.png" --method hybrid`


**Общие параметры для всех методов**
- --brightness или -b: коэффициент усиления яркости (по умолчанию: 1.8)
- --contrast или -c: коэффициент усиления контраста (по умолчанию: 1.5)
- --output или -o: директория для сохранения результатов (по умолчанию: ./out)
- --no-display или -nd: не показывать результаты, только сохранить


## Сравнение методов

`python main.py --image_path "D:\\VKR_BMSTU\\report\\code\\img\\img.png" --compare `



## Исследование

### Сравнение метрик методов в зависимости от уровня состаривания

#### на датасете nyu
`python comparator.py --output-dir ./research_results  --datasets nyu --methods hybrid midas segmentation --aging-levels 3`
#### на датасете lightfield
`python comparator.py --output-dir ./research_results  --datasets lightfield --methods hybrid midas segmentation --aging-levels 3 `


### Сравнение метода с различными способами состаривания в зависимости от уровня состаривания

#### на датасете nyu метод hybrid
`python comparator.py  --comprehensive-aging --aging-dataset nyu --methods hybrid --aging-levels 3`    
#### на датасете lightfield метод hybrid
`python comparator.py  --comprehensive-aging --aging-dataset lightfield --methods hybrid --aging-levels 3`




