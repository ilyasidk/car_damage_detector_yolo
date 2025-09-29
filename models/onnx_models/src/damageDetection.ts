// Основной класс для детекции повреждений автомобилей

import * as ort from 'onnxruntime-web';
import { ImageUtils } from './imageUtils';
import { 
  DetectionBox, 
  DamageResult, 
  ProcessResult, 
  YOLOClass, 
  SeverityClass, 
  ModelConfig 
} from './types';

export class DamageDetectionPipeline {
  private yoloSession: ort.InferenceSession | null = null;
  private classifierSession: ort.InferenceSession | null = null;
  private config: ModelConfig;
  
  private readonly yoloClasses: Record<number, YOLOClass> = {
    0: 'dirt',
    1: 'scratch', 
    2: 'dent'
  };
  
  private readonly severityClasses: Record<number, SeverityClass> = {
    0: 'dirt',
    1: 'scratch_low',
    2: 'scratch_med', 
    3: 'scratch_high',
    4: 'dent_low',
    5: 'dent_med',
    6: 'dent_high'
  };

  constructor(config: ModelConfig) {
    this.config = config;
  }

  /**
   * Инициализация моделей
   */
  async initialize(): Promise<void> {
    try {
      // Настраиваем ONNX Runtime для веба
      ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.16.3/dist/';
      
      // Загружаем модели
      this.yoloSession = await ort.InferenceSession.create(this.config.yoloModelPath);
      this.classifierSession = await ort.InferenceSession.create(this.config.classifierModelPath);
      
      console.log('Модели успешно загружены');
    } catch (error) {
      console.error('Ошибка загрузки моделей:', error);
      throw error;
    }
  }

  /**
   * Детекция повреждений с помощью YOLO
   */
  private async detectDamages(imageData: ImageData): Promise<DetectionBox[]> {
    if (!this.yoloSession) {
      throw new Error('YOLO модель не инициализирована');
    }

    // Подготавливаем изображение для YOLO
    const resized = ImageUtils.resizeImage(imageData, 640, 640);
    const normalized = ImageUtils.normalizeForYOLO(resized);
    
    // Создаем тензор для ONNX
    const inputTensor = new ort.Tensor('float32', normalized, [1, 3, 640, 640]);
    
    // Получаем предсказания
    const outputs = await this.yoloSession.run({ images: inputTensor });
    
    // Обрабатываем результаты (упрощенная версия)
    return this.processYOLOOutputs(outputs, imageData.width, imageData.height);
  }

  /**
   * Обработка выходов YOLO
   */
  private processYOLOOutputs(outputs: ort.InferenceSession.OnnxValueMapType, originalWidth: number, originalHeight: number): DetectionBox[] {
    // Получаем выходы YOLO (обычно это массив [batch, num_detections, 85])
    const predictions = outputs.output || outputs[Object.keys(outputs)[0]];
    const data = predictions.data as Float32Array;
    const shape = predictions.dims;
    
    const detections: DetectionBox[] = [];
    const scaleX = originalWidth / 640;
    const scaleY = originalHeight / 640;
    
    // Обрабатываем каждое обнаружение
    for (let i = 0; i < shape[1]; i++) {
      const startIdx = i * shape[2];
      
      // Извлекаем координаты и уверенность
      const x = data[startIdx] * scaleX;
      const y = data[startIdx + 1] * scaleY;
      const width = data[startIdx + 2] * scaleX;
      const height = data[startIdx + 3] * scaleY;
      
      // Получаем уверенности для каждого класса
      const confidences = [
        data[startIdx + 4], // dirt
        data[startIdx + 5], // scratch  
        data[startIdx + 6]  // dent
      ];
      
      const maxConfidence = Math.max(...confidences);
      const classIndex = confidences.indexOf(maxConfidence);
      
      if (maxConfidence > this.config.confidenceThreshold) {
        detections.push({
          x: x - width / 2,
          y: y - height / 2,
          width,
          height,
          confidence: maxConfidence,
          class: this.yoloClasses[classIndex]
        });
      }
    }
    
    // Применяем NMS
    return this.applyNMS(detections);
  }

  /**
   * Non-Maximum Suppression
   */
  private applyNMS(detections: DetectionBox[]): DetectionBox[] {
    // Сортируем по уверенности
    detections.sort((a, b) => b.confidence - a.confidence);
    
    const filtered: DetectionBox[] = [];
    
    for (const detection of detections) {
      let shouldAdd = true;
      
      for (const existing of filtered) {
        const iou = this.calculateIoU(detection, existing);
        if (iou > this.config.nmsThreshold) {
          shouldAdd = false;
          break;
        }
      }
      
      if (shouldAdd) {
        filtered.push(detection);
      }
    }
    
    return filtered;
  }

  /**
   * Вычисление IoU (Intersection over Union)
   */
  private calculateIoU(box1: DetectionBox, box2: DetectionBox): number {
    const x1 = Math.max(box1.x, box2.x);
    const y1 = Math.max(box1.y, box2.y);
    const x2 = Math.min(box1.x + box1.width, box2.x + box2.width);
    const y2 = Math.min(box1.y + box1.height, box2.y + box2.height);
    
    if (x2 <= x1 || y2 <= y1) return 0;
    
    const intersection = (x2 - x1) * (y2 - y1);
    const area1 = box1.width * box1.height;
    const area2 = box2.width * box2.height;
    const union = area1 + area2 - intersection;
    
    return intersection / union;
  }

  /**
   * Классификация степени тяжести
   */
  private async classifySeverity(cropImageData: ImageData): Promise<{ class: SeverityClass; confidence: number }> {
    if (!this.classifierSession) {
      throw new Error('Классификатор не инициализирован');
    }

    // Подготавливаем изображение для классификатора
    const transformed = ImageUtils.transformForClassifier(cropImageData);
    const inputTensor = new ort.Tensor('float32', transformed, [1, 3, 224, 224]);
    
    // Получаем предсказание
    const outputs = await this.classifierSession.run({ input: inputTensor });
    const logits = outputs.output || outputs[Object.keys(outputs)[0]];
    const probabilities = this.softmax(logits.data as Float32Array);
    
    const maxIndex = probabilities.indexOf(Math.max(...probabilities));
    const confidence = probabilities[maxIndex];
    
    return {
      class: this.severityClasses[maxIndex],
      confidence
    };
  }

  /**
   * Softmax функция
   */
  private softmax(logits: Float32Array): Float32Array {
    const max = Math.max(...logits);
    const exp = logits.map(x => Math.exp(x - max));
    const sum = exp.reduce((a, b) => a + b, 0);
    return exp.map(x => x / sum) as Float32Array;
  }

  /**
   * Полная обработка изображения
   */
  async processImage(imageData: ImageData): Promise<ProcessResult> {
    const startTime = performance.now();
    
    try {
      // Этап 1: Детекция повреждений
      const detections = await this.detectDamages(imageData);
      
      // Этап 2: Классификация степени тяжести для каждого обнаружения
      const results: DamageResult[] = [];
      
      for (const detection of detections) {
        // Извлекаем область повреждения
        const crop = ImageUtils.extractCrop(
          imageData,
          Math.max(0, Math.round(detection.x)),
          Math.max(0, Math.round(detection.y)),
          Math.round(detection.width),
          Math.round(detection.height)
        );
        
        // Классифицируем степень тяжести
        const severity = await this.classifySeverity(crop);
        
        results.push({
          box: detection,
          severityClass: severity.class,
          severityConfidence: severity.confidence
        });
      }
      
      const processingTime = performance.now() - startTime;
      
      return {
        detections: results,
        processingTime,
        imageSize: {
          width: imageData.width,
          height: imageData.height
        }
      };
      
    } catch (error) {
      console.error('Ошибка обработки изображения:', error);
      throw error;
    }
  }

  /**
   * Очистка ресурсов
   */
  dispose(): void {
    this.yoloSession?.release();
    this.classifierSession?.release();
    this.yoloSession = null;
    this.classifierSession = null;
  }
}
