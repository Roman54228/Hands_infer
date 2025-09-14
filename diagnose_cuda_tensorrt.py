#!/usr/bin/env python3
"""
Диагностический скрипт для проверки CUDA и TensorRT совместимости
"""
import os
import sys
import subprocess
import numpy as np

def check_cuda_installation():
    """Проверяет установку CUDA"""
    print("=== Проверка CUDA ===")
    
    try:
        # Проверяем nvidia-smi
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✓ nvidia-smi доступен")
            print("Информация о GPU:")
            print(result.stdout.split('\n')[0:3])  # Первые 3 строки
        else:
            print("✗ nvidia-smi недоступен")
            return False
    except FileNotFoundError:
        print("✗ nvidia-smi не найден")
        return False
    
    try:
        # Проверяем CUDA runtime
        import pycuda.driver as cuda
        import pycuda.autoinit
        print(f"✓ PyCUDA доступен")
        print(f"CUDA версия: {cuda.get_version()}")
        print(f"Количество GPU: {cuda.Device.count()}")
        
        for i in range(cuda.Device.count()):
            dev = cuda.Device(i)
            print(f"GPU {i}: {dev.name()}")
            print(f"  Compute Capability: {dev.compute_capability()}")
            print(f"  Memory: {dev.total_memory() / 1024**3:.1f} GB")
        
        return True
    except ImportError:
        print("✗ PyCUDA не установлен")
        return False
    except Exception as e:
        print(f"✗ Ошибка PyCUDA: {e}")
        return False

def check_tensorrt_installation():
    """Проверяет установку TensorRT"""
    print("\n=== Проверка TensorRT ===")
    
    try:
        import tensorrt as trt
        print(f"✓ TensorRT доступен")
        print(f"TensorRT версия: {trt.__version__}")
        
        # Проверяем доступные плагины
        print("Доступные плагины:")
        for plugin in trt.get_plugin_registry().plugin_creator_list:
            print(f"  - {plugin.name}")
        
        return True
    except ImportError:
        print("✗ TensorRT не установлен")
        return False
    except Exception as e:
        print(f"✗ Ошибка TensorRT: {e}")
        return False

def check_polygraphy():
    """Проверяет установку Polygraphy"""
    print("\n=== Проверка Polygraphy ===")
    
    try:
        from polygraphy.backend.trt import EngineFromBytes, TrtRunner
        print("✓ Polygraphy доступен")
        
        # Проверяем версию
        try:
            import polygraphy
            print(f"Polygraphy версия: {polygraphy.__version__}")
        except:
            print("Версия Polygraphy неизвестна")
        
        return True
    except ImportError:
        print("✗ Polygraphy не установлен")
        return False
    except Exception as e:
        print(f"✗ Ошибка Polygraphy: {e}")
        return False

def test_simple_tensorrt():
    """Тестирует простое создание TensorRT контекста"""
    print("\n=== Тест простого TensorRT ===")
    
    try:
        import tensorrt as trt
        import pycuda.driver as cuda
        import pycuda.autoinit
        
        # Создаем простой logger
        logger = trt.Logger(trt.Logger.WARNING)
        
        # Создаем builder
        builder = trt.Builder(logger)
        print("✓ TensorRT Builder создан")
        
        # Создаем network
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        print("✓ TensorRT Network создан")
        
        # Создаем config
        config = builder.create_builder_config()
        print("✓ TensorRT Config создан")
        
        # Проверяем доступные профили
        print("Доступные профили:")
        for i in range(builder.num_optimization_profiles):
            print(f"  Профиль {i}")
        
        print("✓ Базовый тест TensorRT прошел успешно")
        return True
        
    except Exception as e:
        print(f"✗ Ошибка в базовом тесте TensorRT: {e}")
        return False

def test_cuda_context():
    """Тестирует создание CUDA контекста"""
    print("\n=== Тест CUDA контекста ===")
    
    try:
        import pycuda.driver as cuda
        import pycuda.autoinit
        
        # Создаем простой CUDA контекст
        context = cuda.Context()
        print("✓ CUDA контекст создан")
        
        # Тестируем выделение памяти
        test_array = cuda.mem_alloc(1024)  # 1KB
        print("✓ CUDA память выделена")
        
        # Освобождаем память
        test_array.free()
        print("✓ CUDA память освобождена")
        
        return True
        
    except Exception as e:
        print(f"✗ Ошибка в тесте CUDA контекста: {e}")
        return False

def check_engine_files():
    """Проверяет наличие engine файлов"""
    print("\n=== Проверка Engine файлов ===")
    
    engine_files = [
        "yolo11n.engine",
        "mobilenetv3_hand_cls.engine", 
        "kps.engine"
    ]
    
    all_found = True
    for engine_file in engine_files:
        if os.path.exists(engine_file):
            size = os.path.getsize(engine_file) / 1024**2  # MB
            print(f"✓ {engine_file} найден ({size:.1f} MB)")
        else:
            print(f"✗ {engine_file} не найден")
            all_found = False
    
    return all_found

def main():
    """Основная функция диагностики"""
    print("Диагностика CUDA и TensorRT совместимости")
    print("=" * 50)
    
    results = {
        "CUDA": check_cuda_installation(),
        "TensorRT": check_tensorrt_installation(),
        "Polygraphy": check_polygraphy(),
        "Simple TensorRT": test_simple_tensorrt(),
        "CUDA Context": test_cuda_context(),
        "Engine Files": check_engine_files()
    }
    
    print("\n" + "=" * 50)
    print("РЕЗУЛЬТАТЫ ДИАГНОСТИКИ:")
    print("=" * 50)
    
    all_passed = True
    for test_name, result in results.items():
        status = "✓ ПРОЙДЕН" if result else "✗ ПРОВАЛЕН"
        print(f"{test_name}: {status}")
        if not result:
            all_passed = False
    
    print("\n" + "=" * 50)
    if all_passed:
        print("✓ ВСЕ ТЕСТЫ ПРОЙДЕНЫ")
        print("Проблема может быть в конкретной модели или данных")
    else:
        print("✗ НЕКОТОРЫЕ ТЕСТЫ ПРОВАЛЕНЫ")
        print("Необходимо исправить проблемы перед использованием TensorRT")
    
    print("\nРекомендации:")
    print("1. Убедитесь, что версии CUDA, TensorRT и PyCUDA совместимы")
    print("2. Проверьте, что GPU поддерживает TensorRT")
    print("3. Попробуйте пересоздать engine файлы")
    print("4. Используйте fallback на последовательное выполнение при ошибках")

if __name__ == "__main__":
    main()
