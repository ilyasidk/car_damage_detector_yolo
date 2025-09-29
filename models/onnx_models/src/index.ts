// Главный экспорт модуля

export { DamageDetectionPipeline } from './damageDetection';
export { ImageUtils } from './imageUtils';
export * from './types';

// Удобная функция для быстрого создания пайплайна
export function createDamageDetectionPipeline(
  yoloModelPath: string = './yolo_3class.onnx',
  classifierModelPath: string = './severity_classifier.onnx',
  confidenceThreshold: number = 0.25,
  nmsThreshold: number = 0.45
) {
  return new DamageDetectionPipeline({
    yoloModelPath,
    classifierModelPath,
    confidenceThreshold,
    nmsThreshold
  });
}
