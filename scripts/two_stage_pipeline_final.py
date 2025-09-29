import torch
import cv2
import numpy as np
from ultralytics import YOLO
import onnxruntime as ort
from PIL import Image
import matplotlib.pyplot as plt
import os

class TwoStageDamageDetection:
    def __init__(self, yolo_model_path, classifier_onnx_path):
        """
        Two-stage damage detection system:
        1. YOLO - damage detection (3 classes: dirt, scratch, dent)
        2. Classifier - severity classification (6 classes)
        """
        print("Initializing two-stage system...")
        
        # Load YOLO model
        print("Loading YOLO model...")
        self.yolo_model = YOLO(yolo_model_path)
        
        # Load ONNX classifier
        print("Loading ONNX classifier...")
        self.classifier_session = ort.InferenceSession(classifier_onnx_path)
        
        # YOLO classes (3 classes)
        self.yolo_classes = ['dirt', 'scratch', 'dent']
        
        # Classifier classes (6 classes)
        self.classifier_classes = [
            'dirt', 'scratch_low', 'scratch_med', 'scratch_high', 
            'dent_low', 'dent_med', 'dent_high'
        ]
        
        # Transformations for classifier
        self.transform = self._get_transform()
        
        print("System initialized!")
    
    def _get_transform(self):
        """Returns transformations for classifier"""
        from torchvision import transforms
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def detect_damages(self, image_path, confidence_threshold=0.3):
        """
        Детекция повреждений с помощью YOLO
        
        Args:
            image_path: путь к изображению
            confidence_threshold: порог уверенности для детекции
            
        Returns:
            list: список детекций с координатами и классами
        """
        print(f"🔍 Детекция повреждений в {image_path}...")
        
        # YOLO детекция
        results = self.yolo_model(image_path, conf=confidence_threshold)
        
        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Координаты
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = box.conf[0].cpu().numpy()
                    class_id = int(box.cls[0].cpu().numpy())
                    
                    detection = {
                        'bbox': [int(x1), int(y1), int(x2), int(y2)],
                        'confidence': float(confidence),
                        'yolo_class': self.yolo_classes[class_id],
                        'class_id': class_id
                    }
                    detections.append(detection)
        
        print(f"✅ Найдено {len(detections)} повреждений")
        return detections
    
    def classify_severity(self, image_path, detections):
        """
        Классификация степени тяжести только для scratch и dent
        
        Args:
            image_path: путь к изображению
            detections: список детекций от YOLO
            
        Returns:
            list: список детекций с добавленной информацией о степени тяжести
        """
        if not detections:
            return detections
        
        # Фильтруем только scratch и dent для классификации
        scratch_dent_detections = [d for d in detections if d['yolo_class'] in ['scratch', 'dent']]
        dirt_detections = [d for d in detections if d['yolo_class'] == 'dirt']
        
        print(f"🎯 Классификация степени тяжести для {len(scratch_dent_detections)} повреждений (scratch/dent)...")
        print(f"📝 Пропускаем {len(dirt_detections)} повреждений типа dirt")
        
        enhanced_detections = []
        
        # Обрабатываем dirt - без дополнительной классификации
        for detection in dirt_detections:
            enhanced_detection = detection.copy()
            enhanced_detection['severity_class'] = 'dirt'  # Для dirt оставляем как есть
            enhanced_detection['severity_confidence'] = detection['confidence']  # Используем уверенность YOLO
            enhanced_detections.append(enhanced_detection)
        
        # Обрабатываем scratch и dent - с классификацией
        if scratch_dent_detections:
            # Загружаем изображение
            image = cv2.imread(image_path)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            for detection in scratch_dent_detections:
                # Извлекаем ROI
                x1, y1, x2, y2 = detection['bbox']
                roi = image_rgb[y1:y2, x1:x2]
                
                if roi.size == 0:
                    continue
                
                # Преобразуем ROI для классификатора
                roi_pil = Image.fromarray(roi)
                roi_tensor = self.transform(roi_pil).unsqueeze(0)
                
                # Классификация
                severity_class, severity_confidence = self._classify_roi(roi_tensor)
                
                # Добавляем информацию о степени тяжести
                enhanced_detection = detection.copy()
                enhanced_detection['severity_class'] = severity_class
                enhanced_detection['severity_confidence'] = severity_confidence
                
                enhanced_detections.append(enhanced_detection)
        
        print(f"✅ Классификация завершена для {len(enhanced_detections)} повреждений")
        return enhanced_detections
    
    def _classify_roi(self, roi_tensor):
        """
        Классификация ROI с помощью ONNX модели
        
        Args:
            roi_tensor: тензор ROI (1, 3, 224, 224)
            
        Returns:
            tuple: (класс, уверенность)
        """
        # Конвертируем в numpy
        input_data = roi_tensor.numpy()
        
        # Запускаем ONNX модель
        input_name = self.classifier_session.get_inputs()[0].name
        output_name = self.classifier_session.get_outputs()[0].name
        
        outputs = self.classifier_session.run([output_name], {input_name: input_data})
        predictions = outputs[0][0]
        
        # Получаем класс с максимальной вероятностью
        class_id = np.argmax(predictions)
        confidence = float(predictions[class_id])
        
        return self.classifier_classes[class_id], confidence
    
    def process_image(self, image_path, confidence_threshold=0.3, save_result=True):
        """
        Полная обработка изображения: детекция + классификация
        
        Args:
            image_path: путь к изображению
            confidence_threshold: порог уверенности для YOLO
            save_result: сохранять ли результат
            
        Returns:
            dict: результаты обработки
        """
        print(f"\n🖼️ Обработка изображения: {os.path.basename(image_path)}")
        
        # Этап 1: Детекция повреждений
        detections = self.detect_damages(image_path, confidence_threshold)
        
        if not detections:
            print("❌ Повреждения не найдены")
            return {
                'image_path': image_path,
                'detections': [],
                'summary': 'Повреждения не найдены'
            }
        
        # Этап 2: Классификация степени тяжести
        enhanced_detections = self.classify_severity(image_path, detections)
        
        # Создаем сводку
        summary = self._create_summary(enhanced_detections)
        
        # Сохраняем результат
        if save_result:
            self._save_result(image_path, enhanced_detections, summary)
        
        return {
            'image_path': image_path,
            'detections': enhanced_detections,
            'summary': summary
        }
    
    def _create_summary(self, detections):
        """Создает сводку по детекциям"""
        if not detections:
            return "Повреждения не найдены"
        
        # Группируем по типам
        type_counts = {}
        severity_counts = {}
        
        for det in detections:
            yolo_class = det['yolo_class']
            severity = det['severity_class']
            
            type_counts[yolo_class] = type_counts.get(yolo_class, 0) + 1
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
        
        # Создаем текст сводки
        summary = f"Найдено {len(detections)} повреждений:\n"
        
        summary += "\nПо типам:\n"
        for type_name, count in type_counts.items():
            summary += f"  - {type_name}: {count}\n"
        
        summary += "\nПо степени тяжести:\n"
        for severity, count in severity_counts.items():
            summary += f"  - {severity}: {count}\n"
        
        return summary
    
    def _save_result(self, image_path, detections, summary):
        """Сохраняет результат с визуализацией"""
        # Загружаем изображение
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Рисуем детекции
        for i, det in enumerate(detections):
            x1, y1, x2, y2 = det['bbox']
            yolo_class = det['yolo_class']
            severity = det['severity_class']
            confidence = det['confidence']
            severity_conf = det['severity_confidence']
            
            # Цвет для типа повреждения
            colors = {'dirt': (255, 0, 0), 'scratch': (0, 255, 0), 'dent': (0, 0, 255)}
            color = colors.get(yolo_class, (128, 128, 128))
            
            # Рисуем прямоугольник
            cv2.rectangle(image_rgb, (x1, y1), (x2, y2), color, 2)
            
            # Текст с информацией
            label = f"{yolo_class} ({severity})"
            conf_text = f"YOLO: {confidence:.2f}, Severity: {severity_conf:.2f}"
            
            # Фон для текста
            cv2.rectangle(image_rgb, (x1, y1-40), (x1+len(label)*10, y1), color, -1)
            cv2.rectangle(image_rgb, (x1, y1-20), (x1+len(conf_text)*8, y1-20), color, -1)
            
            # Текст
            cv2.putText(image_rgb, label, (x1, y1-25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(image_rgb, conf_text, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Сохраняем результат
        result_path = f"two_stage_result_{os.path.basename(image_path)}"
        plt.figure(figsize=(12, 8))
        plt.imshow(image_rgb)
        plt.axis('off')
        plt.title(f"Двухэтапная детекция: {os.path.basename(image_path)}")
        plt.tight_layout()
        plt.savefig(result_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        # Сохраняем текстовую сводку
        summary_path = f"two_stage_summary_{os.path.basename(image_path)}.txt"
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(summary)
        
        print(f"💾 Результат сохранен: {result_path}")
        print(f"📝 Сводка сохранена: {summary_path}")

def main():
    """Основная функция для тестирования"""
    print("🚀 Запуск двухэтапной системы детекции повреждений")
    
    # Пути к моделям
    yolo_model_path = "yolo_training_3class_optimized/3class_detection_optimized/weights/best.pt"
    classifier_onnx_path = "onnx_models/severity_classifier.onnx"
    
    # Проверяем наличие файлов
    if not os.path.exists(yolo_model_path):
        print(f"❌ YOLO модель не найдена: {yolo_model_path}")
        return
    
    if not os.path.exists(classifier_onnx_path):
        print(f"❌ ONNX классификатор не найден: {classifier_onnx_path}")
        return
    
    # Создаем систему
    system = TwoStageDamageDetection(yolo_model_path, classifier_onnx_path)
    
    # Тестируем на нескольких изображениях
    test_images = [
        "cars/00010.jpg",
        "cars/00020.jpg", 
        "cars/00030.jpg"
    ]
    
    for image_path in test_images:
        if os.path.exists(image_path):
            print(f"\n{'='*60}")
            result = system.process_image(image_path, confidence_threshold=0.3)
            print(f"\n📊 Результат:")
            print(result['summary'])
        else:
            print(f"⚠️ Изображение не найдено: {image_path}")

if __name__ == "__main__":
    main()
