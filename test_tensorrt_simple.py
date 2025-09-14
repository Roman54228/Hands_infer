#!/usr/bin/env python3
"""
Простой тест для проверки работы TensorRT с обработкой ошибок
"""
import numpy as np
import cv2
import time
from tensorrt_streams import ParallelTensorRTManager


def create_simple_test_image():
    """Создает простое тестовое изображение"""
    # Создаем простое изображение с рукой
    img = np.ones((256, 256, 3), dtype=np.uint8) * 128  # Серый фон
    # Рисуем простую "руку" - круг
    cv2.circle(img, (128, 128), 50, (200, 200, 200), -1)
    return img


def test_simple_inference():
    """Простой тест inference"""
    print("=== Простой тест TensorRT inference ===")
    
    # Создаем тестовые данные
    test_crops = [create_simple_test_image() for _ in range(2)]
    print(f"✓ Создано {len(test_crops)} тестовых изображений")
    
    # Инициализация менеджера
    try:
        print("\nИнициализация ParallelTensorRTManager...")
        manager = ParallelTensorRTManager(
            gesture_model_path="mobilenetv3_hand_cls.engine",
            keypoints_model_path="kps.engine"
        )
        
        if not manager.is_ready():
            print("✗ Менеджер не готов к работе")
            return False
        
        print("✓ Менеджер готов к работе")
        
    except Exception as e:
        print(f"✗ Ошибка инициализации менеджера: {e}")
        return False
    
    # Тест последовательного выполнения
    print("\n--- Тест последовательного выполнения ---")
    try:
        start_time = time.time()
        gestures, keypoints = manager._predict_sequential(test_crops)
        sequential_time = time.time() - start_time
        
        print(f"✓ Последовательное выполнение: {sequential_time:.4f}s")
        print(f"  Жесты: {gestures}")
        print(f"  Ключевые точки: {len(keypoints)} наборов")
        
        if len(gestures) == len(test_crops) and len(keypoints) == len(test_crops):
            print("✓ Результаты корректны")
        else:
            print("⚠ Неожиданное количество результатов")
            
    except Exception as e:
        print(f"✗ Ошибка последовательного выполнения: {e}")
        return False
    
    # Тест параллельного выполнения
    print("\n--- Тест параллельного выполнения ---")
    try:
        start_time = time.time()
        gestures, keypoints = manager.predict_parallel(test_crops)
        parallel_time = time.time() - start_time
        
        print(f"✓ Параллельное выполнение: {parallel_time:.4f}s")
        print(f"  Жесты: {gestures}")
        print(f"  Ключевые точки: {len(keypoints)} наборов")
        
        if len(gestures) == len(test_crops) and len(keypoints) == len(test_crops):
            print("✓ Результаты корректны")
        else:
            print("⚠ Неожиданное количество результатов")
            
    except Exception as e:
        print(f"✗ Ошибка параллельного выполнения: {e}")
        return False
    
    print("\n=== Тест завершен успешно ===")
    return True


def test_error_handling():
    """Тест обработки ошибок"""
    print("\n=== Тест обработки ошибок ===")
    
    # Тест с пустым списком
    try:
        manager = ParallelTensorRTManager(
            gesture_model_path="mobilenetv3_hand_cls.engine",
            keypoints_model_path="kps.engine"
        )
        
        gestures, keypoints = manager.predict_parallel([])
        print("✓ Обработка пустого списка работает")
        
    except Exception as e:
        print(f"✗ Ошибка обработки пустого списка: {e}")
    
    # Тест с несуществующими файлами
    try:
        print("\nТест с несуществующими файлами...")
        manager = ParallelTensorRTManager(
            gesture_model_path="nonexistent_gesture.engine",
            keypoints_model_path="nonexistent_kps.engine"
        )
        
        if not manager.is_ready():
            print("✓ Менеджер корректно определил недоступность моделей")
        else:
            print("⚠ Менеджер считает недоступные модели готовыми")
            
    except Exception as e:
        print(f"✓ Ошибка корректно обработана: {e}")


if __name__ == "__main__":
    print("Тестирование TensorRT с обработкой ошибок")
    print("=" * 50)
    
    success = test_simple_inference()
    
    if success:
        test_error_handling()
        print("\n🎉 Все тесты пройдены!")
    else:
        print("\n❌ Тесты провалены")
        print("\nРекомендации:")
        print("1. Проверьте наличие engine файлов")
        print("2. Убедитесь в совместимости версий CUDA/TensorRT")
        print("3. Запустите diagnose_cuda_tensorrt.py для диагностики")
