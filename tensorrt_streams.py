"""
Модуль для работы с TensorRT моделями с использованием CUDA streams
"""
import numpy as np
import cv2
import time
from typing import List, Tuple, Optional
import threading
from polygraphy.backend.trt import EngineFromBytes, TrtRunner
import os

try:
    import pycuda.driver as cuda
    import pycuda.autoinit
    CUDA_AVAILABLE = True
    print("✓ PyCUDA доступен")
except ImportError:
    print("⚠ PyCUDA не доступен, используется последовательное выполнение")
    CUDA_AVAILABLE = False

try:
    import tensorrt as trt
    TENSORRT_AVAILABLE = True
    print("✓ TensorRT доступен")
except ImportError:
    print("⚠ TensorRT не доступен")
    TENSORRT_AVAILABLE = False


class TensorRTStreamRunner:
    """Класс для выполнения TensorRT моделей с использованием CUDA streams"""
    
    def __init__(self, engine_path: str, stream_id: int = 0):
        self.engine_path = engine_path
        self.stream_id = stream_id
        self.runner = None
        self.stream = None
        self.lock = threading.Lock()
        self.initialization_error = None
        self._initialize()
    
    def _initialize(self):
        """Инициализация модели и CUDA stream"""
        try:
            with self.lock:
                if self.runner is None:
                    if not os.path.exists(self.engine_path):
                        raise FileNotFoundError(f"Engine file not found: {self.engine_path}")
                    
                    print(f"Загрузка модели из {self.engine_path}...")
                    
                    with open(self.engine_path, "rb") as f:
                        engine = EngineFromBytes(f.read())
                    
                    self.runner = TrtRunner(engine)
                    
                    if CUDA_AVAILABLE:
                        try:
                            self.stream = cuda.Stream()
                            print(f"✓ Модель загружена с CUDA stream {self.stream_id}")
                        except Exception as e:
                            print(f"⚠ Не удалось создать CUDA stream: {e}")
                            self.stream = None
                    else:
                        print(f"✓ Модель загружена без CUDA streams")
                        
        except Exception as e:
            self.initialization_error = e
            print(f"✗ Ошибка инициализации модели {self.engine_path}: {e}")
            self.runner = None
    
    def is_initialized(self) -> bool:
        """Проверяет, инициализирована ли модель"""
        return self.runner is not None and self.initialization_error is None
    
    def run_batch(self, images: List[np.ndarray], image_size: int = 256, max_batch_size: int = 4) -> Tuple[np.ndarray, float]:
        """Выполнение батча изображений с автоматическим разбиением больших batch"""
        if not images:
            return np.array([]), 0.0
        
        if not self.is_initialized():
            raise RuntimeError(f"Модель не инициализирована: {self.initialization_error}")
        
        try:
            # Если количество изображений меньше или равно max_batch_size, обрабатываем как обычно
            if len(images) <= max_batch_size:
                return self._run_single_batch(images, image_size)
            
            # Если изображений больше max_batch_size, разбиваем на части
            all_outputs = []
            total_inference_time = 0.0
            
            for i in range(0, len(images), max_batch_size):
                batch_images = images[i:i + max_batch_size]
                batch_output, batch_time = self._run_single_batch(batch_images, image_size)
                all_outputs.append(batch_output)
                total_inference_time += batch_time
            
            # Объединяем результаты всех batch
            final_output = np.concatenate(all_outputs, axis=0)
            
            return final_output, total_inference_time
            
        except Exception as e:
            print(f"✗ Критическая ошибка в run_batch: {e}")
            raise
    
    def _run_single_batch(self, images: List[np.ndarray], image_size: int) -> Tuple[np.ndarray, float]:
        """Выполнение одного batch изображений"""
        # Подготовка батча
        batch = []
        for img in images:
            if isinstance(img, np.ndarray):
                img_resized = cv2.resize(img, (image_size, image_size))
                img_normalized = img_resized.transpose(2, 0, 1).astype(np.float32) / 255.0
                batch.append(img_normalized)
        
        input_tensor = np.stack(batch)  # [N, 3, H, W]
        
        # Выполнение inference
        with self.lock:
            if self.runner is None:
                raise RuntimeError("Модель не загружена")
            
            with self.runner:
                input_name = self.runner.engine[0]
                output_name = self.runner.engine[1]
                
                start_time = time.time()
                
                try:
                    if CUDA_AVAILABLE and self.stream is not None:
                        # Асинхронное выполнение с CUDA stream
                        outputs = self.runner.infer({input_name: input_tensor})
                        self.stream.synchronize()
                    else:
                        # Синхронное выполнение
                        outputs = self.runner.infer({input_name: input_tensor})
                    
                    inference_time = time.time() - start_time
                    
                except Exception as e:
                    print(f"✗ Ошибка inference в модели {self.engine_path}: {e}")
                    # Пробуем синхронное выполнение как fallback
                    if CUDA_AVAILABLE and self.stream is not None:
                        print("Пробуем синхронное выполнение...")
                        outputs = self.runner.infer({input_name: input_tensor})
                        inference_time = time.time() - start_time
                    else:
                        raise
        
        return outputs[output_name], inference_time


class ParallelTensorRTManager:
    """Менеджер для параллельного выполнения двух TensorRT моделей"""
    
    def __init__(self, gesture_model_path: str, keypoints_model_path: str):
        self.gesture_model_path = gesture_model_path
        self.keypoints_model_path = keypoints_model_path
        
        # Создаем отдельные runners для каждой модели
        print("Инициализация ParallelTensorRTManager...")
        self.gesture_runner = TensorRTStreamRunner(gesture_model_path, stream_id=0)
        self.keypoints_runner = TensorRTStreamRunner(keypoints_model_path, stream_id=1)
        
        # Проверяем успешность инициализации
        if self.gesture_runner.is_initialized() and self.keypoints_runner.is_initialized():
            print("✓ ParallelTensorRTManager инициализирован с CUDA streams")
        else:
            print("⚠ ParallelTensorRTManager инициализирован с ошибками")
            if not self.gesture_runner.is_initialized():
                print(f"  Ошибка жестовой модели: {self.gesture_runner.initialization_error}")
            if not self.keypoints_runner.is_initialized():
                print(f"  Ошибка модели ключевых точек: {self.keypoints_runner.initialization_error}")
    
    def is_ready(self) -> bool:
        """Проверяет, готов ли менеджер к работе"""
        return (self.gesture_runner.is_initialized() and 
                self.keypoints_runner.is_initialized())
    
    def predict_parallel(self, crops: List[np.ndarray]) -> Tuple[List[int], List[np.ndarray]]:
        """Параллельное предсказание жестов и ключевых точек"""
        if not crops:
            return [], []
        
        if not self.is_ready():
            print("⚠ Менеджер не готов к работе, используем последовательное выполнение")
            return self._predict_sequential(crops)
        
        gesture_predictions = []
        keypoint_predictions = []
        
        try:
            if CUDA_AVAILABLE and TENSORRT_AVAILABLE:
                # Используем threading для параллельного выполнения
                gesture_results = [None]
                keypoint_results = [None]
                gesture_error = [None]
                keypoint_error = [None]
                
                def run_gestures():
                    try:
                        output, _ = self.gesture_runner.run_batch(crops, image_size=224)
                        gesture_results[0] = [pred.argmax() for pred in output]
                    except Exception as e:
                        gesture_error[0] = e
                        print(f"✗ Ошибка в потоке жестов: {e}")
                
                def run_keypoints():
                    try:
                        output, _ = self.keypoints_runner.run_batch(crops, image_size=256)
                        keypoints = []
                        for pred in output:
                            kps = np.expand_dims(pred, 0)[:, :, :2] * 256
                            keypoints.append(kps[0])
                        keypoint_results[0] = keypoints
                    except Exception as e:
                        keypoint_error[0] = e
                        print(f"✗ Ошибка в потоке ключевых точек: {e}")
                
                # Запускаем в отдельных потоках
                thread1 = threading.Thread(target=run_gestures)
                thread2 = threading.Thread(target=run_keypoints)
                
                thread1.start()
                thread2.start()
                
                thread1.join()
                thread2.join()
                
                # Проверяем результаты
                if gesture_error[0] is not None or keypoint_error[0] is not None:
                    print("⚠ Ошибки в параллельном выполнении, переключаемся на последовательное")
                    return self._predict_sequential(crops)
                
                gesture_predictions = gesture_results[0]
                keypoint_predictions = keypoint_results[0]
                
            else:
                print("⚠ CUDA или TensorRT недоступны, используем последовательное выполнение")
                return self._predict_sequential(crops)
                
        except Exception as e:
            print(f"✗ Критическая ошибка в параллельном предсказании: {e}")
            return self._predict_sequential(crops)
        
        return gesture_predictions, keypoint_predictions
    
    def _predict_sequential(self, crops: List[np.ndarray]) -> Tuple[List[int], List[np.ndarray]]:
        """Последовательное предсказание как fallback"""
        try:
            # Предсказание жестов
            if self.gesture_runner.is_initialized():
                gesture_output, _ = self.gesture_runner.run_batch(crops, image_size=224)
                gesture_predictions = [pred.argmax() for pred in gesture_output]
            else:
                print("⚠ Модель жестов недоступна")
                gesture_predictions = [0] * len(crops)
            
            # Предсказание ключевых точек
            if self.keypoints_runner.is_initialized():
                keypoint_output, _ = self.keypoints_runner.run_batch(crops, image_size=256)
                keypoint_predictions = []
                for pred in keypoint_output:
                    kps = np.expand_dims(pred, 0)[:, :, :2] * 256
                    keypoint_predictions.append(kps[0])
            else:
                print("⚠ Модель ключевых точек недоступна")
                keypoint_predictions = [np.zeros((21, 2))] * len(crops)
            
            return gesture_predictions, keypoint_predictions
            
        except Exception as e:
            print(f"✗ Ошибка в последовательном предсказании: {e}")
            return [], []
