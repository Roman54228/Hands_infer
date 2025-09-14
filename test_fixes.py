#!/usr/bin/env python3
"""
Быстрый тест исправлений диагностического скрипта
"""
import sys
import os

def test_tensorrt_basic():
    """Тестирует базовую функциональность TensorRT"""
    print("=== Тест базовой функциональности TensorRT ===")
    
    try:
        import tensorrt as trt
        print(f"✓ TensorRT {trt.__version__} импортирован")
        
        # Создаем logger
        logger = trt.Logger(trt.Logger.WARNING)
        print("✓ Logger создан")
        
        # Создаем builder
        builder = trt.Builder(logger)
        print("✓ Builder создан")
        
        # Создаем network
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        print("✓ Network создан")
        
        # Создаем config
        config = builder.create_builder_config()
        print("✓ Config создан")
        
        print("✓ Базовый тест TensorRT прошел успешно")
        return True
        
    except Exception as e:
        print(f"✗ Ошибка в базовом тесте TensorRT: {e}")
        return False

def test_cuda_basic():
    """Тестирует базовую функциональность CUDA"""
    print("\n=== Тест базовой функциональности CUDA ===")
    
    try:
        import pycuda.driver as cuda
        import pycuda.autoinit
        print("✓ PyCUDA импортирован")
        
        # Проверяем количество устройств
        device_count = cuda.Device.count()
        print(f"✓ Количество GPU: {device_count}")
        
        if device_count > 0:
            device = cuda.Device(0)
            print(f"✓ GPU 0: {device.name()}")
            
            # Тестируем выделение памяти
            test_array = cuda.mem_alloc(1024)
            print("✓ Память выделена")
            test_array.free()
            print("✓ Память освобождена")
        
        print("✓ Базовый тест CUDA прошел успешно")
        return True
        
    except Exception as e:
        print(f"✗ Ошибка в базовом тесте CUDA: {e}")
        return False

def test_polygraphy_basic():
    """Тестирует базовую функциональность Polygraphy"""
    print("\n=== Тест базовой функциональности Polygraphy ===")
    
    try:
        from polygraphy.backend.trt import EngineFromBytes, TrtRunner
        print("✓ Polygraphy импортирован")
        
        # Проверяем наличие engine файлов
        engine_files = ["mobilenetv3_hand_cls.engine", "kps.engine"]
        found_files = []
        
        for engine_file in engine_files:
            if os.path.exists(engine_file):
                size = os.path.getsize(engine_file) / 1024**2
                print(f"✓ {engine_file} найден ({size:.1f} MB)")
                found_files.append(engine_file)
            else:
                print(f"✗ {engine_file} не найден")
        
        if found_files:
            print(f"✓ Найдено {len(found_files)} engine файлов")
            return True
        else:
            print("⚠ Engine файлы не найдены")
            return False
        
    except Exception as e:
        print(f"✗ Ошибка в базовом тесте Polygraphy: {e}")
        return False

def main():
    """Основная функция"""
    print("Быстрый тест исправлений")
    print("=" * 40)
    
    results = {
        "TensorRT": test_tensorrt_basic(),
        "CUDA": test_cuda_basic(),
        "Polygraphy": test_polygraphy_basic()
    }
    
    print("\n" + "=" * 40)
    print("РЕЗУЛЬТАТЫ:")
    print("=" * 40)
    
    passed = 0
    for test_name, result in results.items():
        status = "✓ ПРОЙДЕН" if result else "✗ ПРОВАЛЕН"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nПройдено: {passed}/{len(results)} тестов")
    
    if passed == len(results):
        print("🎉 Все тесты пройдены! Исправления работают.")
    else:
        print("⚠ Некоторые тесты провалены. Требуется дополнительная настройка.")
    
    print("\nСледующие шаги:")
    print("1. Запустите обновленный diagnose_cuda_tensorrt.py")
    print("2. Если нужно, создайте недостающие engine файлы")
    print("3. Протестируйте с test_tensorrt_simple.py")

if __name__ == "__main__":
    main()
