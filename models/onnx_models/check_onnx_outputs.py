#!/usr/bin/env python3
"""
Скрипт для проверки формата выходных данных ONNX моделей
"""

import onnxruntime as ort
import numpy as np
import cv2

def check_model_outputs(model_path, input_shape, input_name="images"):
    """Проверяет формат выходных данных ONNX модели"""
    print(f"\n=== Анализ модели: {model_path} ===")
    
    try:
        # Загружаем модель
        session = ort.InferenceSession(model_path)
        
        print("Входные данные:")
        for input_meta in session.get_inputs():
            print(f"  Имя: {input_meta.name}")
            print(f"  Тип: {input_meta.type}")
            print(f"  Форма: {input_meta.shape}")
        
        print("\nВыходные данные:")
        for output_meta in session.get_outputs():
            print(f"  Имя: {output_meta.name}")
            print(f"  Тип: {output_meta.type}")
            print(f"  Форма: {output_meta.shape}")
        
        # Создаем тестовый входной тензор
        test_input = np.random.randn(*input_shape).astype(np.float32)
        input_dict = {input_name: test_input}
        
        # Запускаем модель
        outputs = session.run(None, input_dict)
        
        print(f"\nРеальные выходные данные:")
        for i, output in enumerate(outputs):
            print(f"  Выход {i}:")
            print(f"    Форма: {output.shape}")
            print(f"    Тип данных: {output.dtype}")
            print(f"    Диапазон значений: [{output.min():.4f}, {output.max():.4f}]")
            if output.size < 20:  # Показываем только если мало элементов
                print(f"    Значения: {output.flatten()}")
        
        return outputs
        
    except Exception as e:
        print(f"Ошибка: {e}")
        return None

def main():
    print("Проверка формата выходных данных ONNX моделей")
    
    # Проверяем YOLO модель
    yolo_outputs = check_model_outputs(
        "yolo_3class.onnx", 
        (1, 3, 640, 640), 
        "images"
    )
    
    # Проверяем классификатор
    classifier_outputs = check_model_outputs(
        "severity_classifier.onnx", 
        (1, 3, 224, 224), 
        "input"
    )
    
    print("\n=== Сводка ===")
    print("YOLO модель:")
    if yolo_outputs:
        print(f"  Количество выходов: {len(yolo_outputs)}")
        for i, output in enumerate(yolo_outputs):
            print(f"  Выход {i}: {output.shape}")
    
    print("\nКлассификатор:")
    if classifier_outputs:
        print(f"  Количество выходов: {len(classifier_outputs)}")
        for i, output in enumerate(classifier_outputs):
            print(f"  Выход {i}: {output.shape}")

if __name__ == "__main__":
    main()
