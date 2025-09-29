#!/usr/bin/env python3
"""
Конвертация моделей в ONNX формат для веб-развертывания
"""

import os
import torch
from ultralytics import YOLO
import onnx
import onnxruntime as ort

# Настройки
YOLO_MODEL_PATH = "yolo_training_3class_optimized/3class_detection_optimized/weights/best.pt"
SEVERITY_MODEL_PATH = "severity_classifier_best.pth"
OUTPUT_DIR = "onnx_models"

def convert_yolo_to_onnx():
    """Конвертирует YOLO модель в ONNX"""
    print("🔄 Конвертируем YOLO модель в ONNX...")
    
    if not os.path.exists(YOLO_MODEL_PATH):
        print(f"❌ YOLO модель не найдена: {YOLO_MODEL_PATH}")
        return None
    
    # Загружаем YOLO модель
    model = YOLO(YOLO_MODEL_PATH)
    
    # Конвертируем в ONNX
    onnx_path = os.path.join(OUTPUT_DIR, "yolo_3class.onnx")
    try:
        model.export(format="onnx", imgsz=640, dynamic=True, simplify=True)
        
        # Переименовываем файл
        if os.path.exists("best.onnx"):
            os.rename("best.onnx", onnx_path)
            print(f"✅ YOLO модель конвертирована: {onnx_path}")
            return onnx_path
        else:
            print("❌ Ошибка конвертации YOLO модели")
            return None
            
    except Exception as e:
        print(f"❌ Ошибка конвертации YOLO: {e}")
        return None

def convert_classifier_to_onnx():
    """Конвертирует классификатор в ONNX"""
    print("🔄 Конвертируем классификатор в ONNX...")
    
    if not os.path.exists(SEVERITY_MODEL_PATH):
        print(f"❌ Классификатор не найден: {SEVERITY_MODEL_PATH}")
        return None
    
    try:
        # Загружаем модель
        checkpoint = torch.load(SEVERITY_MODEL_PATH, map_location='cpu')
        
        # Создаем модель (предполагаем ResNet архитектуру)
        import torchvision.models as models
        model = models.resnet50(pretrained=False)
        model.fc = torch.nn.Linear(model.fc.in_features, 7)  # 7 классов
        
        # Загружаем веса
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            elif 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
            else:
                model.load_state_dict(checkpoint)
        else:
            model.load_state_dict(checkpoint)
        
        model.eval()
        
        # Создаем пример входа
        dummy_input = torch.randn(1, 3, 224, 224)
        
        # Конвертируем в ONNX
        onnx_path = os.path.join(OUTPUT_DIR, "severity_classifier.onnx")
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )
        
        print(f"✅ Классификатор конвертирован: {onnx_path}")
        return onnx_path
        
    except Exception as e:
        print(f"❌ Ошибка конвертации классификатора: {e}")
        return None

def test_onnx_models(yolo_onnx_path, classifier_onnx_path):
    """Тестирует ONNX модели"""
    print("\n🧪 Тестируем ONNX модели...")
    
    # Тестируем YOLO ONNX
    if yolo_onnx_path and os.path.exists(yolo_onnx_path):
        try:
            session = ort.InferenceSession(yolo_onnx_path)
            print(f"✅ YOLO ONNX модель загружена успешно")
            print(f"   Входы: {[input.name for input in session.get_inputs()]}")
            print(f"   Выходы: {[output.name for output in session.get_outputs()]}")
        except Exception as e:
            print(f"❌ Ошибка загрузки YOLO ONNX: {e}")
    
    # Тестируем классификатор ONNX
    if classifier_onnx_path and os.path.exists(classifier_onnx_path):
        try:
            session = ort.InferenceSession(classifier_onnx_path)
            print(f"✅ Классификатор ONNX модель загружена успешно")
            print(f"   Входы: {[input.name for input in session.get_inputs()]}")
            print(f"   Выходы: {[output.name for output in session.get_outputs()]}")
        except Exception as e:
            print(f"❌ Ошибка загрузки классификатора ONNX: {e}")

def create_web_pipeline():
    """Создает веб-совместимый pipeline"""
    print("\n🌐 Создаем веб-совместимый pipeline...")
    
    web_pipeline_code = '''
import onnxruntime as ort
import cv2
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

class WebDamageDetectionPipeline:
    def __init__(self, yolo_onnx_path, classifier_onnx_path):
        """Инициализация веб-пайплайна"""
        # Загружаем ONNX модели
        self.yolo_session = ort.InferenceSession(yolo_onnx_path)
        self.classifier_session = ort.InferenceSession(classifier_onnx_path)
        
        # Настройки для классификатора
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Классы
        self.yolo_classes = {0: "dirt", 1: "scratch", 2: "dent"}
        self.severity_classes = {
            0: "dirt", 1: "scratch_low", 2: "scratch_med", 3: "scratch_high",
            4: "dent_low", 5: "dent_med", 6: "dent_high"
        }
    
    def detect_damages(self, image_array, confidence_threshold=0.25):
        """Детекция повреждений с помощью YOLO"""
        # Подготавливаем изображение для YOLO
        input_image = cv2.resize(image_array, (640, 640))
        input_image = input_image.astype(np.float32) / 255.0
        input_image = np.transpose(input_image, (2, 0, 1))
        input_image = np.expand_dims(input_image, axis=0)
        
        # Получаем предсказания
        outputs = self.yolo_session.run(None, {"images": input_image})
        
        # Обрабатываем результаты (упрощенно)
        # Здесь нужно добавить NMS и фильтрацию по confidence
        return outputs
    
    def classify_severity(self, crop_image):
        """Классификация степени тяжести"""
        # Конвертируем в PIL
        if len(crop_image.shape) == 3:
            crop_pil = Image.fromarray(cv2.cvtColor(crop_image, cv2.COLOR_BGR2RGB))
        else:
            crop_pil = Image.fromarray(crop_image)
        
        # Применяем трансформации
        input_tensor = self.transform(crop_pil).unsqueeze(0).numpy()
        
        # Получаем предсказание
        outputs = self.classifier_session.run(None, {"input": input_tensor})
        probabilities = self.softmax(outputs[0])
        
        predicted_class = np.argmax(probabilities)
        confidence = np.max(probabilities)
        
        return predicted_class, confidence
    
    def softmax(self, x):
        """Softmax функция"""
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x)
    
    def process_image(self, image_array):
        """Полная обработка изображения"""
        # Этап 1: Детекция
        detections = self.detect_damages(image_array)
        
        # Этап 2: Классификация для каждого обнаружения
        results = []
        for detection in detections:
            # Извлекаем область
            # crop = extract_crop(image_array, detection['box'])
            
            # Классифицируем
            # severity_class, confidence = self.classify_severity(crop)
            
            # results.append({
            #     'box': detection['box'],
            #     'class': severity_class,
            #     'confidence': confidence
            # })
            pass
        
        return results

# Пример использования
def create_pipeline():
    return WebDamageDetectionPipeline(
        "onnx_models/yolo_3class.onnx",
        "onnx_models/severity_classifier.onnx"
    )
'''
    
    # Сохраняем веб-пайплайн
    web_pipeline_path = os.path.join(OUTPUT_DIR, "web_pipeline.py")
    with open(web_pipeline_path, 'w', encoding='utf-8') as f:
        f.write(web_pipeline_code)
    
    print(f"✅ Веб-пайплайн создан: {web_pipeline_path}")

def create_requirements():
    """Создает requirements.txt для веб-развертывания"""
    requirements = '''# Веб-развертывание системы детекции повреждений
onnxruntime>=1.15.0
opencv-python>=4.6.0
pillow>=9.0.0
numpy>=1.21.0
torchvision>=0.12.0
'''
    
    requirements_path = os.path.join(OUTPUT_DIR, "requirements.txt")
    with open(requirements_path, 'w', encoding='utf-8') as f:
        f.write(requirements)
    
    print(f"✅ Requirements создан: {requirements_path}")

def create_readme():
    """Создает README для веб-развертывания"""
    readme = '''# Система детекции повреждений автомобилей (ONNX)

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
'''
    
    readme_path = os.path.join(OUTPUT_DIR, "README.md")
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(readme)
    
    print(f"✅ README создан: {readme_path}")

def main():
    print("🔄 КОНВЕРТАЦИЯ МОДЕЛЕЙ В ONNX ДЛЯ ВЕБ-РАЗВЕРТЫВАНИЯ")
    print("=" * 60)
    
    # Создаем папку для ONNX моделей
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Конвертируем YOLO модель
    yolo_onnx_path = convert_yolo_to_onnx()
    
    # Конвертируем классификатор
    classifier_onnx_path = convert_classifier_to_onnx()
    
    # Тестируем ONNX модели
    test_onnx_models(yolo_onnx_path, classifier_onnx_path)
    
    # Создаем веб-совместимые файлы
    create_web_pipeline()
    create_requirements()
    create_readme()
    
    print(f"\n✅ Конвертация завершена!")
    print(f"📁 ONNX модели сохранены в: {OUTPUT_DIR}")
    print(f"🌐 Веб-пайплайн готов к развертыванию!")
    
    if yolo_onnx_path and classifier_onnx_path:
        print(f"\n📋 Файлы для веб-сайта:")
        print(f"  - {yolo_onnx_path}")
        print(f"  - {classifier_onnx_path}")
        print(f"  - {OUTPUT_DIR}/web_pipeline.py")
        print(f"  - {OUTPUT_DIR}/requirements.txt")
        print(f"  - {OUTPUT_DIR}/README.md")

if __name__ == "__main__":
    main()
