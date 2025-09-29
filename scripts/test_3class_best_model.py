#!/usr/bin/env python3
"""
Тестирование лучшей модели 3-классовой детекции
"""

import os
import random
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import matplotlib.pyplot as plt

# Настройки
MODEL_PATH = "yolo_training_3class_optimized/3class_detection_optimized/weights/best.pt"
TEST_IMAGES_DIR = "yolo_data_3class/test/images"
OUTPUT_DIR = "test_3class_best_model"
NUM_TEST_IMAGES = 20

# Классы
CLASS_NAMES = {
    0: "dirt",
    1: "scratch", 
    2: "dent"
}

# Цвета для визуализации
CLASS_COLORS = {
    0: (255, 0, 0),      # dirt - красный
    1: (0, 255, 0),      # scratch - зеленый
    2: (0, 0, 255)       # dent - синий
}

def test_model():
    """Тестирование модели на случайных изображениях"""
    print("🔍 ТЕСТИРОВАНИЕ ЛУЧШЕЙ 3-КЛАССОВОЙ МОДЕЛИ")
    print("=" * 50)
    
    # Проверяем модель
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Модель не найдена: {MODEL_PATH}")
        return
    
    # Загружаем модель
    print(f"📥 Загружаем модель: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)
    print("✅ Модель загружена")
    
    # Создаем папку для результатов
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Получаем тестовые изображения
    test_images = list(Path(TEST_IMAGES_DIR).glob("*.jpg"))
    if not test_images:
        print(f"❌ Тестовые изображения не найдены в {TEST_IMAGES_DIR}")
        return
    
    print(f"📸 Найдено {len(test_images)} тестовых изображений")
    
    # Выбираем случайные изображения
    selected_images = random.sample(test_images, min(NUM_TEST_IMAGES, len(test_images)))
    print(f"🎲 Выбрано {len(selected_images)} изображений для тестирования")
    
    # Статистика
    total_detections = 0
    class_detections = {0: 0, 1: 0, 2: 0}
    confidence_scores = []
    
    # Тестируем каждое изображение
    for i, image_path in enumerate(selected_images):
        print(f"\n📸 [{i+1}/{len(selected_images)}] Тестируем: {image_path.name}")
        
        # Получаем предсказания
        results = model(str(image_path), conf=0.25, iou=0.5)
        
        # Загружаем изображение
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"  ❌ Не удалось загрузить изображение")
            continue
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width = image_rgb.shape[:2]
        
        # Анализируем результаты
        detections = 0
        if results[0].boxes is not None and len(results[0].boxes) > 0:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            confidences = results[0].boxes.conf.cpu().numpy()
            classes = results[0].boxes.cls.cpu().numpy().astype(int)
            
            detections = len(boxes)
            total_detections += detections
            
            print(f"  🎯 Найдено объектов: {detections}")
            
            # Рисуем bounding boxes
            for box, conf, cls in zip(boxes, confidences, classes):
                x1, y1, x2, y2 = box.astype(int)
                class_name = CLASS_NAMES.get(cls, f"unknown_{cls}")
                color = CLASS_COLORS.get(cls, (255, 255, 255))
                
                # Подсчитываем статистику
                class_detections[cls] += 1
                confidence_scores.append(conf)
                
                print(f"    - {class_name}: {conf:.3f} [{x1},{y1},{x2},{y2}]")
                
                # Рисуем прямоугольник
                cv2.rectangle(image_rgb, (x1, y1), (x2, y2), color, 3)
                
                # Подпись
                label = f"{class_name}: {conf:.2f}"
                font_scale = max(0.5, min(1.0, width / 1000))
                thickness = max(1, int(width / 500))
                
                # Размер текста
                (text_width, text_height), baseline = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
                )
                
                # Фон для текста
                cv2.rectangle(image_rgb, 
                             (x1, y1 - text_height - 10), 
                             (x1 + text_width, y1), 
                             color, -1)
                
                # Текст
                cv2.putText(image_rgb, label, (x1, y1 - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)
        else:
            print(f"  ❌ Объекты не найдены")
        
        # Сохраняем результат
        output_path = os.path.join(OUTPUT_DIR, f"test_{i+1:02d}_{image_path.stem}.jpg")
        cv2.imwrite(output_path, cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
        print(f"  💾 Сохранено: {output_path}")
    
    # Выводим статистику
    print(f"\n📊 СТАТИСТИКА ТЕСТИРОВАНИЯ:")
    print(f"  - Всего изображений: {len(selected_images)}")
    print(f"  - Всего детекций: {total_detections}")
    print(f"  - Среднее детекций на изображение: {total_detections / len(selected_images):.2f}")
    
    if confidence_scores:
        print(f"  - Средняя уверенность: {np.mean(confidence_scores):.3f}")
        print(f"  - Медианная уверенность: {np.median(confidence_scores):.3f}")
        print(f"  - Мин уверенность: {np.min(confidence_scores):.3f}")
        print(f"  - Макс уверенность: {np.max(confidence_scores):.3f}")
    
    print(f"\n📈 ДЕТЕКЦИИ ПО КЛАССАМ:")
    for class_id, class_name in CLASS_NAMES.items():
        count = class_detections[class_id]
        percentage = count / total_detections * 100 if total_detections > 0 else 0
        print(f"  - {class_name}: {count} ({percentage:.1f}%)")
    
    print(f"\n✅ Тестирование завершено!")
    print(f"📁 Результаты сохранены в: {OUTPUT_DIR}")

def test_on_specific_images():
    """Тестирование на конкретных изображениях с разными классами"""
    print(f"\n🎯 ТЕСТИРОВАНИЕ НА КОНКРЕТНЫХ ИЗОБРАЖЕНИЯХ")
    print("=" * 50)
    
    # Загружаем модель
    model = YOLO(MODEL_PATH)
    
    # Ищем изображения с разными классами
    class_examples = {0: [], 1: [], 2: []}
    
    for split in ['train', 'val', 'test']:
        labels_dir = f"yolo_data_3class/{split}/labels"
        images_dir = f"yolo_data_3class/{split}/images"
        
        if not os.path.exists(labels_dir):
            continue
        
        for label_file in Path(labels_dir).glob("*.txt"):
            try:
                with open(label_file, 'r') as f:
                    lines = f.readlines()
                
                image_classes = set()
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    if len(parts) == 5:
                        class_id = int(parts[0])
                        if class_id in CLASS_NAMES:
                            image_classes.add(class_id)
                
                # Добавляем примеры для каждого класса
                for class_id in image_classes:
                    if len(class_examples[class_id]) < 3:  # По 3 примера на класс
                        image_name = label_file.stem + '.jpg'
                        image_path = Path(images_dir) / image_name
                        if image_path.exists():
                            class_examples[class_id].append(image_path)
                            
            except Exception as e:
                continue
    
    # Тестируем примеры
    for class_id, class_name in CLASS_NAMES.items():
        print(f"\n🔍 Тестируем примеры класса '{class_name}':")
        
        for i, image_path in enumerate(class_examples[class_id]):
            print(f"  [{i+1}] {image_path.name}")
            
            # Получаем предсказания
            results = model(str(image_path), conf=0.25, iou=0.5)
            
            # Анализируем результаты
            if results[0].boxes is not None and len(results[0].boxes) > 0:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                confidences = results[0].boxes.conf.cpu().numpy()
                classes = results[0].boxes.cls.cpu().numpy().astype(int)
                
                print(f"    Найдено: {len(boxes)} объектов")
                for box, conf, cls in zip(boxes, confidences, classes):
                    predicted_class = CLASS_NAMES.get(cls, f"unknown_{cls}")
                    print(f"      - {predicted_class}: {conf:.3f}")
            else:
                print(f"    ❌ Объекты не найдены")

def main():
    # Устанавливаем seed для воспроизводимости
    random.seed(42)
    
    # Основное тестирование
    test_model()
    
    # Тестирование на конкретных примерах
    test_on_specific_images()

if __name__ == "__main__":
    main()
