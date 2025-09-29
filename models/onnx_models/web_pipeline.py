
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
