"""
Модуль для работы с моделями нейросетей
"""
import numpy as np
import cv2
from typing import List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import time
import threading
import queue
import multiprocessing as mp
from ultralytics import YOLO
from run_trt import load_hand_gesture_model, predict_hand_gesture
from new_speed_bench import run_model_batch
from tensorrt_streams import ParallelTensorRTManager


def predict_gestures_process(crops: List[np.ndarray], model_path: str, image_size: int = 256) -> List[int]:
    """Функция для предсказания жестов в отдельном процессе"""
    try:
        print('LOADING MODEL')
        model = load_hand_gesture_model(model_path)
        print('LOADING MODEL2')

        output = run_model_batch(crops, model, image_size=image_size, max_batch_size=4)
        predictions = [pred.argmax() for pred in output[0]]
        return predictions
    except Exception as e:
        print(f"Error in gesture process: {e}")
        return []


def predict_keypoints_process(crops: List[np.ndarray], model_path: str, image_size: int = 256) -> List[np.ndarray]:
    """Функция для предсказания ключевых точек в отдельном процессе"""
    try:
        model = load_hand_gesture_model(model_path)
        output = run_model_batch(crops, model, image_size=image_size, max_batch_size=4)
        keypoints = []
        for pred in output[0]:
            # Нормализация координат
            kps = np.expand_dims(pred, 0)[:, :, :2] * image_size
            keypoints.append(kps[0])
        return keypoints
    except Exception as e:
        print(f"Error in keypoints process: {e}")
        return []


class ThreadSafeTensorRTModel:
    """Thread-safe обертка для TensorRT модели"""
    
    def __init__(self, engine_path: str):
        self.engine_path = engine_path
        self.model = None
        self.lock = threading.Lock()
        self._load_model()
    
    def _load_model(self):
        """Загрузка модели в отдельном потоке"""
        with self.lock:
            if self.model is None:
                self.model = load_hand_gesture_model(self.engine_path)
    
    def predict(self, crops: List[np.ndarray], image_size: int = 256):
        """Thread-safe предсказание"""
        with self.lock:
            if self.model is None:
                self._load_model()
            return run_model_batch(crops, self.model, image_size, max_batch_size=4)


class ModelManager:
    """Класс для управления моделями нейросетей"""
    
    def __init__(self, yolo_model_path: str, hand_gesture_model_path: str, kps_model_path: str, 
                 use_parallel: bool = True, parallel_workers: int = 2):
        self.yolo_model = None
        self.hand_gesture_model = None
        self.kps_model = None
        self.crop_size = 256
        self.use_parallel = use_parallel
        self.parallel_workers = parallel_workers
        
        # Сохраняем пути к моделям для использования в процессах
        self.hand_gesture_model_path = hand_gesture_model_path
        self.kps_model_path = kps_model_path
        
        # Thread-safe модели для параллельного выполнения
        self.thread_safe_gesture_model = None
        self.thread_safe_kps_model = None
        
        # CUDA streams менеджер для параллельного выполнения
        self.parallel_manager = None
        
        # Цвета для разных классов жестов
        self.gesture_colors = {
            0: (255, 0, 0),      # 2_hands
            1: (255, 128, 0),   # fist
            2: (102, 255, 255), # normal_hand
            3: (153, 0, 153),   # pinch
            4: (178, 102, 55)   # pointer
        }
        
        self._load_models(yolo_model_path, hand_gesture_model_path, kps_model_path)
    
    def _load_models(self, yolo_path: str, hand_gesture_path: str, kps_path: str):
        """Загрузка всех моделей"""
        try:
            print("Loading YOLO model...")
            self.yolo_model = YOLO(yolo_path)
            
            print("Loading hand gesture model...")
            self.hand_gesture_model = load_hand_gesture_model(hand_gesture_path)
            
            print("Loading keypoints model...")
            self.kps_model = load_hand_gesture_model(kps_path)
            # self.kps_model = load_hand_gesture_model('mm_kps0.engine')
            # Создаем CUDA streams менеджер для параллельного выполнения
            if self.use_parallel:
                print("Creating CUDA streams manager for parallel execution...")
                try:
                    self.parallel_manager = ParallelTensorRTManager(hand_gesture_path, kps_path)
                    print("✓ CUDA streams manager created successfully")
                except Exception as e:
                    print(f"⚠ Failed to create CUDA streams manager: {e}")
                    print("Falling back to sequential execution")
                    self.use_parallel = False
            
            print("All models loaded successfully!")
            
        except Exception as e:
            print(f"Error loading models: {e}")
            raise
    
    def detect_hands(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Детекция рук с помощью YOLO"""
        if self.yolo_model is None:
            return []
        
        results = self.yolo_model.predict(image)
        detections = []
        
        for i, box in enumerate(results[0].boxes.xyxy):
            x1, y1, x2, y2 = map(int, box.tolist())
            pad = 15
            # x1 -= pad
            # x2 += pad
            # y1 -= pad
            # y2 += pad
            detections.append((x1, y1, x2, y2))
        
        return detections
    
    def crop_hands(self, image: np.ndarray, detections: List[Tuple[int, int, int, int]]) -> List[np.ndarray]:
        """Обрезка изображений рук"""
        crops = []
        for x1, y1, x2, y2 in detections:
            cropped = image[y1:y2, x1:x2]
            if cropped.size > 0:
                resized = cv2.resize(cropped, (self.crop_size, self.crop_size))
                crops.append(resized)
        
        return crops
    
    def predict_gestures(self, crops: List[np.ndarray], image_size) -> List[int]:
        """Предсказание жестов для обрезанных изображений рук"""
        if not crops or self.hand_gesture_model is None:
            return []
        
        try:
            output = run_model_batch(crops, self.hand_gesture_model, image_size=image_size, max_batch_size=4)
            predictions = [pred.argmax() for pred in output[0]]
            return predictions
        except Exception as e:
            print(f"Error predicting gestures: {e}")
            return []
    
    def predict_keypoints(self, crops: List[np.ndarray], image_size) -> List[np.ndarray]:
        """Предсказание ключевых точек рук"""
        if not crops or self.kps_model is None:
            return []
        
        try:
            output = run_model_batch(crops, self.kps_model, image_size=image_size, max_batch_size=4)
            keypoints = []
            for pred in output[0]:
                # Нормализация координат
                kps = np.expand_dims(pred, 0)[:, :, :2] * image_size
                keypoints.append(kps[0])
            return keypoints
        except Exception as e:
            print(f"Error predicting keypoints: {e}")
            return []
    
    def predict_gestures_and_keypoints_parallel(self, crops: List[np.ndarray]) -> Tuple[List[int], List[np.ndarray]]:
        """Параллельное предсказание жестов и ключевых точек с использованием процессов"""
        if not crops:
            return [], []
        
        gesture_predictions = []
        keypoint_predictions = []
        
        try:
            # Используем ProcessPoolExecutor для избежания конфликтов CUDA контекстов
            breakpoint()
            with ProcessPoolExecutor(max_workers=2) as executor:
                # Запускаем обе модели параллельно в разных процессах
                future_gestures = executor.submit(
                    predict_gestures_process, 
                    crops, 
                    self.hand_gesture_model_path, 
                    224
                )
                breakpoint()
                future_keypoints = executor.submit(
                    predict_keypoints_process, 
                    crops, 
                    self.kps_model_path, 
                    256
                )
                
                # Ждем результаты
                gesture_predictions = future_gestures.result()
                keypoint_predictions = future_keypoints.result()
                
        except Exception as e:
            print(f"Error in parallel prediction: {e}")
            print("Falling back to sequential execution...")
            # Fallback к последовательному выполнению
            gesture_predictions = self.predict_gestures(crops, 256)
            keypoint_predictions = self.predict_keypoints(crops, 256)
        
        return gesture_predictions, keypoint_predictions
    
    def predict_gestures_and_keypoints_cuda_streams(self, crops: List[np.ndarray]) -> Tuple[List[int], List[np.ndarray]]:
        """Параллельное предсказание с использованием CUDA streams"""
        if not crops:
            return [], []
        breakpoint()
        if self.parallel_manager is None:
            print("CUDA streams manager not available, falling back to sequential execution")
            gestures = self.predict_gestures(crops, image_size=224)
            keypoints = self.predict_keypoints(crops, image_size=256)
            return gestures, keypoints
        
        try:
            return self.parallel_manager.predict_parallel(crops)
        except Exception as e:
            print(f"Error in CUDA streams prediction: {e}")
            print("Falling back to sequential execution...")
            gestures = self.predict_gestures(crops, image_size=256)
            keypoints = self.predict_keypoints(crops, image_size=256)
            return gestures, keypoints
    
    def predict_gestures_and_keypoints_parallel_threads(self, crops: List[np.ndarray]) -> Tuple[List[int], List[np.ndarray]]:
        """Параллельное предсказание жестов и ключевых точек с использованием потоков (альтернативный метод)"""
        if not crops:
            return [], []
        
        if self.hand_gesture_model is None or self.kps_model is None:
            return [], []
        
        gesture_predictions = []
        keypoint_predictions = []
        
        try:
            # Используем ThreadPoolExecutor с thread-safe моделями
            with ThreadPoolExecutor(max_workers=2) as executor:
                # Запускаем обе модели параллельно
                future_gestures = executor.submit(self._predict_gestures_thread_safe, crops, 256)
                future_keypoints = executor.submit(self._predict_keypoints_thread_safe, crops, 256)
                
                # Ждем результаты
                gesture_predictions = future_gestures.result()
                keypoint_predictions = future_keypoints.result()
                
        except Exception as e:
            print(f"Error in parallel thread prediction: {e}")
            # Fallback к последовательному выполнению
            gesture_predictions = self.predict_gestures(crops, 256)
            keypoint_predictions = self.predict_keypoints(crops, 256)
        
        return gesture_predictions, keypoint_predictions
    
    def _predict_gestures_thread_safe(self, crops: List[np.ndarray], image_size: int) -> List[int]:
        """Thread-safe предсказание жестов"""
        if self.thread_safe_gesture_model is None:
            return self.predict_gestures(crops, image_size)
        return self.thread_safe_gesture_model.predict(crops, image_size)
    
    def _predict_keypoints_thread_safe(self, crops: List[np.ndarray], image_size: int) -> List[np.ndarray]:
        """Thread-safe предсказание ключевых точек"""
        if self.thread_safe_kps_model is None:
            return self.predict_keypoints(crops, image_size)
        return self.thread_safe_kps_model.predict(crops, image_size)
    
    def predict_gestures_and_keypoints(self, crops: List[np.ndarray]) -> Tuple[List[int], List[np.ndarray]]:
        """Универсальный метод для предсказания жестов и ключевых точек"""
        # breakpoint()
        if self.use_parallel and self.parallel_manager is not None:
            # Используем CUDA streams для параллельного выполнения
            # return self.predict_gestures_and_keypoints_cuda_streams(crops)
            return self.predict_gestures_and_keypoints_parallel(crops)
        else:
            # Последовательное выполнение (надежное)
            gestures = self.predict_gestures(crops, image_size=224)
            keypoints = self.predict_keypoints(crops, image_size=256)
            # gestures, keypoints, _ = run_model_batch(crops, self.kps_model, image_size=256, max_batch_size=4)
            # kps_list = []
            # gestures = [pred.argmax() for pred in gestures] 
            # for pred in keypoints:
            #     # Нормализация координат
            #     kps = np.expand_dims(pred, 0)[:, :, :2] * 256
            #     kps_list.append(kps[0])
            # # cls_output, kps_preds = output[0], output[1]
            return gestures, keypoints
    
    def predict_gestures_and_keypoints_optimized(self, crops: List[np.ndarray]) -> Tuple[List[int], List[np.ndarray]]:
        """Оптимизированное предсказание с батчингом (альтернатива параллельному выполнению)"""
        if not crops:
            return [], []
        
        # Используем батчинг для ускорения с учетом ограничений TensorRT
        batch_size = min(4, len(crops))  # Ограничиваем размер батча до 4 для совместимости с TensorRT профилями
        
        gesture_predictions = []
        keypoint_predictions = []
        
        # Обрабатываем батчами
        for i in range(0, len(crops), batch_size):
            batch_crops = crops[i:i + batch_size]
            
            # Предсказание жестов
            try:
                output_gestures = run_model_batch(batch_crops, self.hand_gesture_model, image_size=256, max_batch_size=4)
                batch_gestures = [pred.argmax() for pred in output_gestures[0]]
                gesture_predictions.extend(batch_gestures)
            except Exception as e:
                print(f"Error in gesture batch {i//batch_size}: {e}")
                gesture_predictions.extend([0] * len(batch_crops))  # Fallback
            
            # Предсказание ключевых точек
            try:
                output_keypoints = run_model_batch(batch_crops, self.kps_model, image_size=256, max_batch_size=4)
                batch_keypoints = []
                for pred in output_keypoints[0]:
                    kps = np.expand_dims(pred, 0)[:, :, :2] * 256
                    batch_keypoints.append(kps[0])
                keypoint_predictions.extend(batch_keypoints)
            except Exception as e:
                print(f"Error in keypoints batch {i//batch_size}: {e}")
                keypoint_predictions.extend([np.zeros((21, 2))] * len(batch_crops))  # Fallback
        
        return gesture_predictions, keypoint_predictions
    
    def _predict_gestures_batch(self, crops: List[np.ndarray], image_size) -> List[int]:
        """Внутренний метод для предсказания жестов"""
        try:
            output = run_model_batch(crops, self.hand_gesture_model, image_size=image_size, max_batch_size=4)
            predictions = [pred.argmax() for pred in output[0]]
            return predictions
        except Exception as e:
            print(f"Error in gesture batch prediction: {e}")
            return []
    
    def _predict_keypoints_batch(self, crops: List[np.ndarray]) -> List[np.ndarray]:
        """Внутренний метод для предсказания ключевых точек"""
        try:
            output = run_model_batch(crops, self.kps_model, image_size=self.crop_size, max_batch_size=4)
            keypoints = []
            for pred in output[0]:
                # Нормализация координат
                kps = np.expand_dims(pred, 0)[:, :, :2] * self.crop_size
                keypoints.append(kps[0])
            return keypoints
        except Exception as e:
            print(f"Error in keypoints batch prediction: {e}")
            return []
    
    def benchmark_parallel_vs_sequential(self, crops: List[np.ndarray], iterations: int = 10) -> dict:
        """Бенчмарк сравнения параллельного и последовательного инференса"""
        if not crops:
            return {"error": "No crops provided"}
        
        results = {
            "parallel_times": [],
            "sequential_times": [],
            "crops_count": len(crops)
        }
        
        print(f"Running benchmark with {iterations} iterations on {len(crops)} crops...")
        
        # Тестируем последовательный инференс
        for i in range(iterations):
            start_time = time.time()
            gestures = self.predict_gestures(crops)
            keypoints = self.predict_keypoints(crops)
            sequential_time = time.time() - start_time
            results["sequential_times"].append(sequential_time)
        
        # Тестируем параллельный инференс
        for i in range(iterations):
            start_time = time.time()
            gestures, keypoints = self.predict_gestures_and_keypoints_parallel(crops)
            parallel_time = time.time() - start_time
            results["parallel_times"].append(parallel_time)
        
        # Вычисляем средние значения
        results["avg_sequential"] = sum(results["sequential_times"]) / len(results["sequential_times"])
        results["avg_parallel"] = sum(results["parallel_times"]) / len(results["parallel_times"])
        results["speedup"] = results["avg_sequential"] / results["avg_parallel"]
        
        print(f"Benchmark Results:")
        print(f"  Sequential average: {results['avg_sequential']:.4f}s")
        print(f"  Parallel average: {results['avg_parallel']:.4f}s")
        print(f"  Speedup: {results['speedup']:.2f}x")
        
        return results
    
    def get_gesture_color(self, gesture_id: int) -> Tuple[int, int, int]:
        """Получение цвета для жеста"""
        return self.gesture_colors.get(gesture_id, (255, 255, 255))
    
    def draw_keypoints(self, image: np.ndarray, keypoints: np.ndarray, 
                      bbox: Tuple[int, int, int, int], color: Tuple[int, int, int]) -> np.ndarray:
        """Отрисовка ключевых точек на изображении"""
        x1, y1, x2, y2 = bbox
        h, w = y2 - y1, x2 - x1
        
        # Соединения между точками
        connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),
            (5, 6), (6, 7), (7, 8),
            (9, 10), (10, 11), (11, 12),
            (13, 14), (14, 15), (15, 16),
            (17, 18), (18, 19), (19, 20),
            (0, 5), (5, 9), (9, 13), (13, 17), (0, 17)
        ]
        
        points = []
        for p_id, (x_norm, y_norm) in enumerate(keypoints):
            x_norm, y_norm = x_norm / self.crop_size, y_norm / self.crop_size
            px = int(x_norm * w)
            py = int(y_norm * h)
            abs_px = x1 + px
            abs_py = y1 + py
            
            points.append((abs_px, abs_py))
            
            # Особые точки (кончики пальцев)
            if p_id in [8, 4]:
                cv2.circle(image, (abs_px, abs_py), 3, (117, 0, 178), -1)
            else:
                cv2.circle(image, (abs_px, abs_py), 1, (0, 255, 0), -1)
        
        # Отрисовка соединений
        for start_idx, end_idx in connections:
            start = points[start_idx]
            end = points[end_idx]
            cv2.line(image, start, end, (0, 255, 0), thickness=1)
        
        return image
