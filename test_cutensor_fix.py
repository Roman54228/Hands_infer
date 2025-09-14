#!/usr/bin/env python3
"""
Тест исправления cuTensor ошибок
"""
import numpy as np
import cv2
from model_manager import ModelManager

def create_test_crops(num_crops: int = 2) -> list:
    """Создает тестовые изображения рук"""
    crops = []
    for i in range(num_crops):
        # Создаем изображение с "рукой" - простой круг
        img = np.ones((256, 256, 3), dtype=np.uint8) * 128  # Серый фон
        cv2.circle(img, (128, 128), 50, (200, 200, 200), -1)  # Белый круг
        crops.append(img)
    return crops

def test_cutensor_fix():
    """Тестирует исправление cuTensor ошибок"""
    print("=== Тест исправления cuTensor ошибок ===")
    
    try:
        # Инициализация ModelManager
        print("Инициализация ModelManager...")
        model_manager = ModelManager(
            yolo_model_path="upside.engine",
            hand_gesture_model_path="best_cls_kps.engine",
            kps_model_path="kps.engine",
            use_parallel=True,
            parallel_workers=2
        )
        
        print("✓ ModelManager инициализирован")
        
        # Проверяем состояние параллельного менеджера
        if model_manager.parallel_manager is not None:
            if model_manager.parallel_manager.is_ready():
                print("✓ CUDA streams менеджер готов к работе")
            else:
                print("⚠ CUDA streams менеджер не готов, будет использоваться fallback")
        else:
            print("⚠ CUDA streams менеджер не создан, будет использоваться последовательное выполнение")
        
        # Тест с небольшим количеством изображений
        test_crops = create_test_crops(2)
        print(f"Тестирование с {len(test_crops)} изображениями...")
        
        gestures, keypoints = model_manager.predict_gestures_and_keypoints(test_crops)
        
        print(f"✓ Успешно обработано {len(test_crops)} изображений!")
        print(f"  Получено жестов: {len(gestures)}")
        print(f"  Получено ключевых точек: {len(keypoints)}")
        print(f"  Жесты: {gestures}")
        
        if len(gestures) == len(test_crops) and len(keypoints) == len(test_crops):
            print("🎉 cuTensor ошибки исправлены!")
            return True
        else:
            print("⚠ Неожиданное количество результатов")
            return False
            
    except Exception as e:
        print(f"✗ Ошибка: {e}")
        if "cuTensor" in str(e) or "CuTensor" in str(e):
            print("⚠ Все еще есть cuTensor ошибки")
        return False

if __name__ == "__main__":
    success = test_cutensor_fix()
    if success:
        print("\n✅ cuTensor ошибки исправлены!")
    else:
        print("\n❌ Требуется дополнительная настройка")
