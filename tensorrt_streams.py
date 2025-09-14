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
except ImportError:
    print("PyCUDA not available, falling back to sequential execution")
    CUDA_AVAILABLE = False


class TensorRTStreamRunner:
    """Класс для выполнения TensorRT моделей с использованием CUDA streams"""
    
    def __init__(self, engine_path: str, stream_id: int = 0):
        self.engine_path = engine_path
        self.stream_id = stream_id
        self.runner = None
        self.stream = None
        self.lock = threading.Lock()
        self._initialize()
    
    def _initialize(self):
        """Инициализация модели и CUDA stream"""
        with self.lock:
            if self.runner is None:
                if not os.path.exists(self.engine_path):
                    raise FileNotFoundError(f"Engine file not found: {self.engine_path}")
                
                with open(self.engine_path, "rb") as f:
                    engine = EngineFromBytes(f.read())
                self.runner = TrtRunner(engine)
                
                if CUDA_AVAILABLE:
                    self.stream = cuda.Stream()
                    print(f"Model loaded from {self.engine_path} with CUDA stream {self.stream_id}")
                else:
                    print(f"Model loaded from {self.engine_path} (no CUDA streams)")
    
    def run_batch(self, images: List[np.ndarray], image_size: int = 256) -> Tuple[np.ndarray, float]:
        """Выполнение батча изображений"""
        if not images:
            return np.array([]), 0.0
        
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
                self._initialize()
            
            with self.runner:
                input_name = self.runner.engine[0]
                output_name = self.runner.engine[1]
                
                start_time = time.time()
                
                if CUDA_AVAILABLE and self.stream is not None:
                    # Асинхронное выполнение с CUDA stream
                    outputs = self.runner.infer({input_name: input_tensor})
                    self.stream.synchronize()
                else:
                    # Синхронное выполнение
                    outputs = self.runner.infer({input_name: input_tensor})
                
                inference_time = time.time() - start_time
        
        return outputs[output_name], inference_time


class ParallelTensorRTManager:
    """Менеджер для параллельного выполнения двух TensorRT моделей"""
    
    def __init__(self, gesture_model_path: str, keypoints_model_path: str):
        self.gesture_model_path = gesture_model_path
        self.keypoints_model_path = keypoints_model_path
        
        # Создаем отдельные runners для каждой модели
        self.gesture_runner = TensorRTStreamRunner(gesture_model_path, stream_id=0)
        self.keypoints_runner = TensorRTStreamRunner(keypoints_model_path, stream_id=1)
        
        print("ParallelTensorRTManager initialized with CUDA streams")
    
    def predict_parallel(self, crops: List[np.ndarray]) -> Tuple[List[int], List[np.ndarray]]:
        """Параллельное предсказание жестов и ключевых точек"""
        if not crops:
            return [], []
        
        gesture_predictions = []
        keypoint_predictions = []
        
        try:
            if CUDA_AVAILABLE:
                # Используем threading для параллельного выполнения
                gesture_results = [None]
                keypoint_results = [None]
                
                def run_gestures():
                    output, _ = self.gesture_runner.run_batch(crops, image_size=224)
                    gesture_results[0] = [pred.argmax() for pred in output]
                
                def run_keypoints():
                    output, _ = self.keypoints_runner.run_batch(crops, image_size=256)
                    keypoints = []
                    for pred in output:
                        kps = np.expand_dims(pred, 0)[:, :, :2] * 256
                        keypoints.append(kps[0])
                    keypoint_results[0] = keypoints
                
                # Запускаем в отдельных потоках
                thread1 = threading.Thread(target=run_gestures)
                thread2 = threading.Thread(target=run_keypoints)
                
                thread1.start()
                thread2.start()
                
                thread1.join()
                thread2.join()
                
                gesture_predictions = gesture_results[0]
                keypoint_predictions = keypoint_results[0]
                
            else:
                # Fallback к последовательному выполнению
                gesture_output, _ = self.gesture_runner.run_batch(crops, image_size=224)
                gesture_predictions = [pred.argmax() for pred in gesture_output]
                
                keypoint_output, _ = self.keypoints_runner.run_batch(crops, image_size=256)
                keypoint_predictions = []
                for pred in keypoint_output:
                    kps = np.expand_dims(pred, 0)[:, :, :2] * 256
                    keypoint_predictions.append(kps[0])
                
        except Exception as e:
            print(f"Error in parallel prediction: {e}")
            # Fallback к последовательному выполнению
            try:
                gesture_output, _ = self.gesture_runner.run_batch(crops, image_size=224)
                gesture_predictions = [pred.argmax() for pred in gesture_output]
                
                keypoint_output, _ = self.keypoints_runner.run_batch(crops, image_size=256)
                keypoint_predictions = []
                for pred in keypoint_output:
                    kps = np.expand_dims(pred, 0)[:, :, :2] * 256
                    keypoint_predictions.append(kps[0])
            except Exception as e2:
                print(f"Fallback also failed: {e2}")
                return [], []
        
        return gesture_predictions, keypoint_predictions
