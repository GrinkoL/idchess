# Задача о локализации
## Описание задачи
На фото с шахматной доской найти координаты четырёх точек её ограничивающих.

![image](https://user-images.githubusercontent.com/74366128/136165508-af763aae-2d65-4548-90cc-38b8b02fe96d.png)
## Решение задачи
Решение предлагается во фреймворке Tensorflow 2.6 + Keras на основе нейросетевой модели MobileNetV2 предобученной на imagenet (fine-tuning подход).
Решение реализовано в рамках одного Colab ноутбука idchess_zadanie.ipynb. 

При компиляции модели использовалась лосс-функция MES. В качестве метрики использовалась метрика IoU.

Датасет для обучения доступен по ссылке:
https://drive.google.com/uc?id=1PXoRm0E9BT1Zw7_D4MMWwKH8sE4NQSXp

## Результат
На выбранной архитектуре, поэкспериментировав с гиперпораметрами удалось достич значения метрики meanIoU равной 92.22% на валидационном датасете.
