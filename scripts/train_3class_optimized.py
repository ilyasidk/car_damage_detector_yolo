#!/usr/bin/env python3
"""
Optimized YOLO training for RTX 3050 Ti 4GB
"""

import os
import random
import torch
import numpy as np
from ultralytics import YOLO

# Исправление для Windows multiprocessing
if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()

# Settings
DATA_YAML = "data/yolo_data_3class/data.yaml"
BASE_MODEL = "YOLOv8s.pt"  # YOLOv8s for 4GB GPU
RUN_DIR = "models/yolo_training_3class_optimized"
RUN_NAME = "3class_detection_optimized"

def setup_seed(seed: int = 42):
    """Set seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def check_gpu_memory():
    """Проверка памяти GPU"""
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"🔍 GPU память: {gpu_memory:.1f} GB")
        
        if gpu_memory < 6:
            print("⚠️ Мало GPU памяти, используем оптимизированные настройки")
            return False
        else:
            print("✅ Достаточно GPU памяти")
            return True
    return False

def check_environment():
    """Проверка окружения"""
    print("🔍 ПРОВЕРКА ОКРУЖЕНИЯ")
    print("=" * 40)
    
    # Проверяем CUDA
    if torch.cuda.is_available():
        print(f"✅ CUDA доступна: {torch.cuda.get_device_name(0)}")
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"   Память GPU: {gpu_memory:.1f} GB")
    else:
        print("⚠️ CUDA недоступна, будет использоваться CPU")
    
    # Проверяем датасет
    if os.path.exists(DATA_YAML):
        print(f"✅ Датасет найден: {DATA_YAML}")
    else:
        print(f"❌ Датасет не найден: {DATA_YAML}")
        return False
    
    # Проверяем базовую модель
    if os.path.exists(BASE_MODEL):
        print(f"✅ Базовая модель найдена: {BASE_MODEL}")
    else:
        print(f"❌ Базовая модель не найдена: {BASE_MODEL}")
        return False
    
    print("=" * 40)
    return True

def check_existing_training():
    """Проверка существующего обучения"""
    run_path = os.path.join(RUN_DIR, RUN_NAME)
    last_pt = os.path.join(run_path, 'weights', 'last.pt')
    
    if os.path.exists(last_pt):
        print(f"🔄 Найдено прерванное обучение: {last_pt}")
        return last_pt
    return None

def train_model():
    """Обучение модели"""
    print("🚀 НАЧИНАЕМ ОПТИМИЗИРОВАННОЕ ОБУЧЕНИЕ")
    print("=" * 40)
    
    # Проверяем память GPU
    has_enough_memory = check_gpu_memory()
    
    # Проверяем, есть ли прерванное обучение
    existing_model = check_existing_training()
    
    if existing_model:
        print(f"📥 Возобновляем обучение с: {existing_model}")
        model = YOLO(existing_model)
        resume_training = True
    else:
        print(f"📥 Загружаем базовую модель: {BASE_MODEL}")
        model = YOLO(BASE_MODEL)
        resume_training = False
    
    # Оптимизированные настройки для RTX 3050 Ti 4GB
    if has_enough_memory:
        # Настройки для GPU с достаточной памятью
        training_args = {
            'data': DATA_YAML,
            'epochs': 150,
            'imgsz': 640,
            'batch': 8,
            'lr0': 0.01,
            'lrf': 0.1,
            'momentum': 0.937,
            'weight_decay': 0.0005,
            'warmup_epochs': 3,
            'warmup_momentum': 0.8,
            'warmup_bias_lr': 0.1,
            'box': 7.5,
            'cls': 0.5,
            'dfl': 1.5,
            'label_smoothing': 0.0,
            'nbs': 64,
            'overlap_mask': True,
            'mask_ratio': 4,
            'dropout': 0.0,
            'val': True,
            'plots': True,
            'save': True,
            'save_period': 10,
            'cache': True,
            'device': '',
            'workers': 4,
            'project': RUN_DIR,
            'name': RUN_NAME,
            'exist_ok': True,
            'pretrained': True,
            'optimizer': 'auto',
            'verbose': True,
            'seed': 42,
            'deterministic': True,
            'single_cls': False,
            'rect': False,
            'cos_lr': False,
            'close_mosaic': 10,
            'resume': resume_training,
            'amp': True,
            'fraction': 1.0,
            'profile': False,
            'freeze': None,
            'multi_scale': False,
            'overlap_mask': True,
            'mask_ratio': 4,
            'dropout': 0.0,
            'val': True,
            'split': 'val',
            'save_json': False,
            'save_hybrid': False,
            'conf': None,
            'iou': 0.7,
            'max_det': 300,
            'half': False,
            'dnn': False,
            'vid_stride': 1
        }
    else:
        # Настройки для GPU с ограниченной памятью
        training_args = {
            'data': DATA_YAML,
            'epochs': 100,
            'imgsz': 640,
            'batch': 2,  # Очень маленький batch
            'lr0': 0.01,
            'lrf': 0.1,
            'momentum': 0.937,
            'weight_decay': 0.0005,
            'warmup_epochs': 3,
            'warmup_momentum': 0.8,
            'warmup_bias_lr': 0.1,
            'box': 7.5,
            'cls': 0.5,
            'dfl': 1.5,
            'label_smoothing': 0.0,
            'nbs': 64,
            'overlap_mask': True,
            'mask_ratio': 4,
            'dropout': 0.0,
            'val': True,
            'plots': True,
            'save': True,
            'save_period': 10,
            'cache': False,  # Отключаем кеш для экономии памяти
            'device': '',
            'workers': 2,  # Меньше воркеров
            'project': RUN_DIR,
            'name': RUN_NAME,
            'exist_ok': True,
            'pretrained': True,
            'optimizer': 'auto',
            'verbose': True,
            'seed': 42,
            'deterministic': True,
            'single_cls': False,
            'rect': False,
            'cos_lr': False,
            'close_mosaic': 10,
            'resume': resume_training,
            'amp': True,
            'fraction': 1.0,
            'profile': False,
            'freeze': None,
            'multi_scale': False,
            'overlap_mask': True,
            'mask_ratio': 4,
            'dropout': 0.0,
            'val': True,
            'split': 'val',
            'save_json': False,
            'save_hybrid': False,
            'conf': None,
            'iou': 0.7,
            'max_det': 300,
            'half': False,
            'dnn': False,
            'vid_stride': 1
        }
    
    print("📋 ОПТИМИЗИРОВАННЫЕ ПАРАМЕТРЫ:")
    print(f"   Модель: {BASE_MODEL}")
    print(f"   Эпохи: {training_args['epochs']}")
    print(f"   Размер изображения: {training_args['imgsz']}")
    print(f"   Batch size: {training_args['batch']}")
    print(f"   Learning rate: {training_args['lr0']}")
    print(f"   Кеш: {training_args['cache']}")
    print(f"   Воркеры: {training_args['workers']}")
    
    # Начинаем обучение
    print("\n🎯 ЗАПУСКАЕМ ОБУЧЕНИЕ...")
    results = model.train(**training_args)
    
    print("✅ ОБУЧЕНИЕ ЗАВЕРШЕНО!")
    return results

def estimate_training_time():
    """Оценка времени обучения"""
    print("\n⏱️ ОЦЕНКА ВРЕМЕНИ ОБУЧЕНИЯ:")
    print("=" * 40)
    
    # Параметры
    images = 1500
    epochs = 100
    batch_size = 2  # Для 4GB GPU
    imgsz = 640
    
    # Расчеты
    images_per_epoch = images * 0.8  # train split
    batches_per_epoch = images_per_epoch / batch_size
    
    # Время на батч для RTX 3050 Ti 4GB
    time_per_batch = 1.5  # секунды
    time_per_epoch = batches_per_epoch * time_per_batch / 60  # минуты
    total_time_hours = (time_per_epoch * epochs) / 60
    
    print(f"Изображений: {images}")
    print(f"Эпох: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Размер изображения: {imgsz}x{imgsz}")
    print(f"Батчей на эпоху: {batches_per_epoch:.0f}")
    print(f"Время на батч: ~{time_per_batch} сек")
    print(f"Время на эпоху: ~{time_per_epoch:.1f} мин")
    print(f"Общее время: ~{total_time_hours:.1f} часов")
    
    return total_time_hours

def main():
    """Основная функция"""
    print("🚀 ОПТИМИЗИРОВАННОЕ ОБУЧЕНИЕ YOLO")
    print("=" * 50)
    print("Модель: YOLOv11s (оптимизированная)")
    print("Классы: dirt, scratch, dent")
    print("GPU: RTX 3050 Ti 4GB")
    print("=" * 50)
    
    # Оценка времени
    estimate_training_time()
    
    # Устанавливаем seed
    setup_seed(42)
    
    # Проверяем окружение
    if not check_environment():
        print("❌ Проблемы с окружением, обучение отменено")
        return
    
    try:
        # Обучаем модель
        results = train_model()
        
        print("\n🎉 ОБУЧЕНИЕ ЗАВЕРШЕНО!")
        print(f"📁 Результаты сохранены в: {os.path.abspath(RUN_DIR)}")
        print(f"🏆 Лучшая модель: {os.path.join(RUN_DIR, RUN_NAME, 'weights', 'best.pt')}")
        
    except KeyboardInterrupt:
        print("\n⏹️ ОБУЧЕНИЕ ОСТАНОВЛЕНО ПОЛЬЗОВАТЕЛЕМ")
        print("💾 Модель сохранена, можно возобновить позже")
        print(f"🔄 Для возобновления запустите скрипт снова")
        
    except Exception as e:
        print(f"❌ Ошибка при обучении: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
