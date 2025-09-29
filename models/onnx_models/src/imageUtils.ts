// Утилиты для работы с изображениями

export class ImageUtils {
  /**
   * Конвертирует HTMLImageElement в ImageData
   */
  static imageToImageData(image: HTMLImageElement): ImageData {
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d')!;
    
    canvas.width = image.width;
    canvas.height = image.height;
    
    ctx.drawImage(image, 0, 0);
    return ctx.getImageData(0, 0, image.width, image.height);
  }

  /**
   * Конвертирует File в ImageData
   */
  static async fileToImageData(file: File): Promise<ImageData> {
    return new Promise((resolve, reject) => {
      const img = new Image();
      img.onload = () => {
        resolve(this.imageToImageData(img));
      };
      img.onerror = reject;
      img.src = URL.createObjectURL(file);
    });
  }

  /**
   * Конвертирует ImageData в Uint8Array для ONNX
   */
  static imageDataToUint8Array(imageData: ImageData): Uint8Array {
    return new Uint8Array(imageData.data);
  }

  /**
   * Изменяет размер изображения
   */
  static resizeImage(imageData: ImageData, width: number, height: number): ImageData {
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d')!;
    
    canvas.width = width;
    canvas.height = height;
    
    // Создаем временный canvas для исходного изображения
    const tempCanvas = document.createElement('canvas');
    const tempCtx = tempCanvas.getContext('2d')!;
    tempCanvas.width = imageData.width;
    tempCanvas.height = imageData.height;
    tempCtx.putImageData(imageData, 0, 0);
    
    // Рисуем с изменением размера
    ctx.drawImage(tempCanvas, 0, 0, width, height);
    
    return ctx.getImageData(0, 0, width, height);
  }

  /**
   * Нормализует изображение для YOLO (0-1 диапазон)
   */
  static normalizeForYOLO(imageData: ImageData): Float32Array {
    const data = new Float32Array(imageData.width * imageData.height * 3);
    const pixels = imageData.data;
    
    for (let i = 0; i < pixels.length; i += 4) {
      const pixelIndex = i / 4;
      data[pixelIndex] = pixels[i] / 255.0;         // R
      data[pixelIndex + imageData.width * imageData.height] = pixels[i + 1] / 255.0;     // G
      data[pixelIndex + 2 * imageData.width * imageData.height] = pixels[i + 2] / 255.0; // B
    }
    
    return data;
  }

  /**
   * Извлекает область изображения по координатам
   */
  static extractCrop(imageData: ImageData, x: number, y: number, width: number, height: number): ImageData {
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d')!;
    
    canvas.width = width;
    canvas.height = height;
    
    const tempCanvas = document.createElement('canvas');
    const tempCtx = tempCanvas.getContext('2d')!;
    tempCanvas.width = imageData.width;
    tempCanvas.height = imageData.height;
    tempCtx.putImageData(imageData, 0, 0);
    
    ctx.drawImage(tempCanvas, x, y, width, height, 0, 0, width, height);
    
    return ctx.getImageData(0, 0, width, height);
  }

  /**
   * Применяет трансформации для классификатора (как в PyTorch)
   */
  static transformForClassifier(imageData: ImageData): Float32Array {
    // Изменяем размер до 224x224
    const resized = this.resizeImage(imageData, 224, 224);
    
    // Нормализуем по ImageNet статистикам
    const data = new Float32Array(3 * 224 * 224);
    const pixels = resized.data;
    
    const mean = [0.485, 0.456, 0.406];
    const std = [0.229, 0.224, 0.225];
    
    for (let i = 0; i < pixels.length; i += 4) {
      const pixelIndex = Math.floor(i / 4);
      const r = (pixels[i] / 255.0 - mean[0]) / std[0];
      const g = (pixels[i + 1] / 255.0 - mean[1]) / std[1];
      const b = (pixels[i + 2] / 255.0 - mean[2]) / std[2];
      
      data[pixelIndex] = r;
      data[pixelIndex + 224 * 224] = g;
      data[pixelIndex + 2 * 224 * 224] = b;
    }
    
    return data;
  }
}
