# Использование
Запуск с различными методами реконструкции

## Метод сегментации (по умолчанию)

`python main.py "E:\\VKR_BMSTU\\report\\code\\img\\testimg5.png" --method segmentation`

Параметры:
- segments или -s: количество цветовых сегментов (по умолчанию: 8)
- iterations или -i: количество итераций для восстановления глубины (по умолчанию: 200)

## Метод MiDaS

`python main.py "E:\\VKR_BMSTU\\report\\code\\img\\testimg5.png" --method midas`

Параметры:
- midas-model: модель MiDaS для использования (по умолчанию: MiDaS_small)
    - MiDaS_small: быстрая работа, но менее точная
    - DPT_Hybrid: баланс между скоростью и качеством
    - DPT_Large: лучшее качество, но более медленная работа

## Гибридный метод (MiDaS + сегментация)

`python main.py "E:\\VKR_BMSTU\\report\\code\\img\\testimg5.png" --method hybrid`


**Общие параметры для всех методов**
- --brightness или -b: коэффициент усиления яркости (по умолчанию: 1.8)
- --contrast или -c: коэффициент усиления контраста (по умолчанию: 1.5)
- --output или -o: директория для сохранения результатов (по умолчанию: ./out)
- --no-display или -nd: не показывать результаты, только сохранить


## Сравнение методов

`python main.py "E:\\VKR_BMSTU\\report\\code\\img\\testimg5.png" --compare `