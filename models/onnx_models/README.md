# Система детекции повреждений автомобилей (ONNX)

## Описание
Двухэтапная система детекции и классификации повреждений:
1. YOLO - детекция областей повреждений (3 класса)
2. Severity Classifier - определение степени тяжести (7 классов)

## Файлы
- `yolo_3class.onnx` - YOLO модель для детекции
- `severity_classifier.onnx` - классификатор степени тяжести
- `web_pipeline.py` - веб-совместимый пайплайн
- `requirements.txt` - зависимости

## Установка
```bash
pip install -r requirements.txt
```

## Использование
```python
from web_pipeline import create_pipeline
import cv2

# Создаем пайплайн
pipeline = create_pipeline()

# Загружаем изображение
image = cv2.imread("test_image.jpg")

# Обрабатываем
results = pipeline.process_image(image)
print(results)
```

## Классы
### YOLO (3 класса):
- dirt - грязные области
- scratch - царапины
- dent - вмятины

### Severity Classifier (7 классов):
- dirt - грязные области
- scratch_low/med/high - степень царапин
- dent_low/med/high - степень вмятин

## Производительность
- YOLO: ~10ms на изображение
- Classifier: ~5ms на область
- Общая скорость: ~15-20ms на изображение
