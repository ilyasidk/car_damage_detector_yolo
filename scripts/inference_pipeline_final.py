#!/usr/bin/env python3
"""
YOLOv8s training for binary defect detection (1 class: damage).
Images without defects should have 0 boxes.
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

DATA_YAML = "../yolo_data_damage/data.yaml"
BASE_MODEL = "../yolov8s.pt"           # <- единообразно используем s
RUN_DIR    = "../yolo_training_damage"
RUN_NAME   = "damage_detection"

# Порог для "бинарного" решения на инференсе
BIN_CONF = 0.25   # можно подкрутить после калибровки PR-кривой

def setup_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # детерминизм может замедлить, но убирает «дрожание»
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def test_environment():
    print("🔧 Тестирование окружения...")
    try:
        import ultralytics
        print(f"✅ Ultralytics: {ultralytics.__version__}")
    except ImportError:
        print("❌ Ultralytics не установлен!")
        return False

    try:
        import torch
        print(f"✅ PyTorch: {torch.__version__}")
    except ImportError:
        print("❌ PyTorch не установлен!")
        return False

    if os.path.exists(DATA_YAML):
        print("✅ data.yaml найден")
    else:
        print("❌ data.yaml не найден!")
        print("💡 Сначала запустите: python scripts/prepare_yolo_damage.py")
        return False

    if os.path.exists(BASE_MODEL):
        print("✅ Базовая модель найдена:", BASE_MODEL)
    else:
        print("⚠️ Базовая модель не найдена, будет загружена автоматически:", BASE_MODEL)

    return True

def main():
    setup_seed(42)

    print("Обучение YOLO модели для бинарной детекции повреждений")
    print("damage - есть повреждения, no_damage - нет повреждений (0 боксов)")
    print("=" * 60)

    print(f"PyTorch версия: {torch.__version__}")
    print(f"CUDA доступна: {torch.cuda.is_available()}")
    device = 0 if torch.cuda.is_available() else 'cpu'
    print(f"Устройство: {'CUDA:0' if device == 0 else 'CPU'}")

    if not os.path.exists(DATA_YAML):
        print("❌ Файл data.yaml не найден!")
        print("💡 Сначала запустите: python scripts/prepare_yolo_damage.py")
        return

    os.makedirs(RUN_DIR, exist_ok=True)

    print("📥 Загружаем базовую модель YOLOv8s...")
    model = YOLO(BASE_MODEL)

    # Профиль «чуть медленнее, но лучше на мелких дефектах»
    epochs = 60
    imgsz = 832        # 640 быстро, 832/960 лучше на царапинах
    batch = 16 if device != 'cpu' else 4

    print("⚙️ Настройки обучения:")
    print(f"   - Эпохи: {epochs}")
    print(f"   - Batch size: {batch}")
    print(f"   - Image size: {imgsz}")
    print("   - Аугментации: умеренные + закрываем mosaic в финале")
    print("   - multi_scale: ON (масштаб помогает мелким объектам)")
    print("   - close_mosaic: 10 (последние 10 эпох без mosaic)")
    print("   - patience: 15 (ранняя остановка)")
    print("   - workers:", 1 if device != 'cpu' else 0)

    print("\n🚀 Начинаем обучение...")
    try:
        results = model.train(
            data=DATA_YAML,
            epochs=epochs,
            batch=batch,
            imgsz=imgsz,
            device=device,
            workers=1 if device != 'cpu' else 0,  # уменьшено для Windows multiprocessing
            lr0=0.001,
            patience=15,
            project=RUN_DIR,
            name=RUN_NAME,
            exist_ok=True,
            save=True,
            save_period=5,
            plots=True,
            cache=(device != 'cpu'),
            amp=(device != 'cpu'),
            # Аугментации (чуть мягче под мелкие дефекты)
            hsv_h=0.015,
            hsv_s=0.5,
            hsv_v=0.4,
            degrees=0.0,
            translate=0.08,
            scale=0.5,
            shear=0.0,
            perspective=0.0,
            flipud=0.0,
            fliplr=0.5,
            mosaic=0.6,        # не 1.0
            close_mosaic=10,   # отключим в конце
            mixup=0.0,
            copy_paste=0.0,
            multi_scale=True,  # важненько
            rect=False,        # для train обычно False; для val можно True
        )

        print("\n✅ Обучение завершено!")
        print(f"📁 Результаты: {os.path.join(RUN_DIR, RUN_NAME)}")

        # Валидация на test-сплите (если определён в data.yaml)
        print("\n🧪 Тестируем модель (split='test' если есть)...")
        test_results = model.val(
            data=DATA_YAML,
            split='test',       # если нет test — Ultralytics возьмёт val
            imgsz=imgsz,
            iou=0.6,
            conf=0.001,         # низкий conf для PR-кривой
            rect=True,          # быстрая валид. укладка
            plots=True,
            save_json=False,
            augment=False       # без TTA для честного сравнения
        )

        print("\n📊 Результаты тестирования:")
        print(f"   - mAP50:    {test_results.box.map50:.3f}")
        print(f"   - mAP50-95: {test_results.box.map:.3f}")
        print(f"   - Precision:{test_results.box.mp:.3f}")
        print(f"   - Recall:   {test_results.box.mr:.3f}")
        f1 = 2 * test_results.box.mp * test_results.box.mr / (test_results.box.mp + test_results.box.mr + 1e-9)
        print(f"   - F1:       {f1:.3f}")

        if test_results.box.map50 > 0.70:
            print("🎉 Отлично! Можно выносить в прод.")
        elif test_results.box.map50 > 0.50:
            print("✅ Хорошо! Ещё немного данных/эпох — и будет топ.")
        elif test_results.box.map50 > 0.30:
            print("⚠️ Средне. Подумайте о разметке/аугах/размере картинок.")
        else:
            print("❌ Слабо. Проверьте баланс классов, разметку и аугментации.")

        # Сохранить и экспортировать
        best_pt = os.path.join(RUN_DIR, RUN_NAME, "weights", "best.pt")
        if os.path.exists(best_pt):
            print(f"\n💾 Лучшая модель: {best_pt}")
            import shutil
            shutil.copy2(best_pt, "../damage_detector.pt")
            print("📋 Копия: ../damage_detector.pt")

            print("📦 Экспорт в ONNX (для прод-инференса)...")
            export_path = model.export(format="onnx", imgsz=imgsz, dynamic=True, simplify=True)
            print("✅ Экспортировано:", export_path)

        print("\n🎉 Готово! Модель обучена и готова к использованию.")

    except Exception as e:
        print(f"❌ Ошибка при обучении: {e}")
        print("💡 Попробуйте:")
        print("   1) batch=1-2")
        print("   2) imgsz=640")
        print("   3) cache=False, amp=False")
        print("   4) проверить разметку и data.yaml")

if __name__ == "__main__":
    if test_environment():
        main()
    else:
        print("\n❌ Исправьте ошибки окружения перед запуском!")
        print("💡 Установите зависимости: pip install 'ultralytics>=8.0.0' torch torchvision")
