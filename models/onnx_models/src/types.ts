// Типы для системы детекции повреждений

export interface DetectionBox {
  x: number;
  y: number;
  width: number;
  height: number;
  confidence: number;
  class: string;
}

export interface DamageResult {
  box: DetectionBox;
  severityClass: string;
  severityConfidence: number;
}

export interface ProcessResult {
  detections: DamageResult[];
  processingTime: number;
  imageSize: {
    width: number;
    height: number;
  };
}

export type YOLOClass = 'dirt' | 'scratch' | 'dent';
export type SeverityClass = 'dirt' | 'scratch_low' | 'scratch_med' | 'scratch_high' | 'dent_low' | 'dent_med' | 'dent_high';

export interface ModelConfig {
  yoloModelPath: string;
  classifierModelPath: string;
  confidenceThreshold: number;
  nmsThreshold: number;
}
