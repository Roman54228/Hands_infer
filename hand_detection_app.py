"""
Основной класс приложения для детекции рук
"""
import cv2
import numpy as np
from typing import Optional
import os

from camera_handler import CameraHandler
from model_manager import ModelManager
from detection_processor import DetectionProcessor
from coordinate_transformer import CoordinateTransformer


class HandDetectionApp:
    """Основной класс приложения для детекции рук"""
    
    def __init__(self, 
                 yolo_model_path: str,
                 hand_gesture_model_path: str,
                 kps_model_path: str,
                 output_dir: str = 'roma_images',
                 osc_ip: str = "10.0.0.101",
                 osc_port: int = 5055,
                 stream_index: int = 1):
        
        # Инициализация компонентов
        self.camera = CameraHandler(stream_index)
        self.models = ModelManager(
            yolo_model_path, 
            hand_gesture_model_path, 
            kps_model_path,
            use_parallel=False,  # Включить параллельный инференс
            parallel_workers=2  # Количество потоков
        )
        self.detector = DetectionProcessor(osc_ip, osc_port)
        self.coord_transformer = CoordinateTransformer()
        self.output_dir = output_dir
        
        # Создание директории для сохранения изображений
        self._create_output_directory()
        
        # Флаг для остановки приложения
        self.running = False
    
    def _create_output_directory(self):
        """Создание директории для сохранения отладочных изображений"""
        try:
            os.makedirs(self.output_dir, exist_ok=True)
        except Exception as e:
            print(f"Warning: Could not create output directory: {e}")
    
    def initialize(self) -> bool:
        """Инициализация всех компонентов"""
        print("Initializing Hand Detection Application...")
        
        # Инициализация камеры
        if not self.camera.initialize():
            print("Failed to initialize camera")
            return False
        
        if not self.camera.setup_stream():
            print("Failed to setup camera stream")
            return False
        
        print("Application initialized successfully!")
        return True
    
    def run(self):
        """Основной цикл приложения"""
        if not self.initialize():
            return
        
        self.running = True
        print("Starting hand detection...")
        print("Press 'q' to quit")
        
        try:
            with self.camera.stream:
                while self.running:
                    # Получение кадра
                    frame = self.camera.get_frame()
                    if frame is None:
                        continue
                    
                    # Обработка только каждого N-го кадра для производительности
                    if not self.detector.should_process_frame():
                        continue
                    
                    # Обработка кадра
                    processed_frame = self._process_frame(frame)
                    
                                      
                    # Обработка клавиш
                    if self._handle_keyboard_input():
                        break
        
        except KeyboardInterrupt:
            print("\nApplication interrupted by user")
        except Exception as e:
            print(f"Error in main loop: {e}")
        finally:
            self.cleanup()
    
    def _process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Обработка одного кадра"""
        draw_image = frame.copy()
        
        # Детекция рук
        detections = self.models.detect_hands(frame)
        if not detections:
            return draw_image
        
        # Ограничиваем количество детекций для производительности
        detections = detections[:4]
        
        # Обрезка изображений рук
        crops = self.models.crop_hands(frame, detections)
        if not crops:
            return draw_image
        # breakpoint()
        # Предсказание жестов и ключевых точек (параллельно или последовательно)
        gesture_predictions, keypoint_predictions = self.models.predict_gestures_and_keypoints(crops)
        
        # Обработка каждой детекции
        for i, bbox in enumerate(detections):
            if i >= len(gesture_predictions) or i >= len(keypoint_predictions):
                continue
            
            x1, y1, x2, y2 = bbox
            gesture_id = gesture_predictions[i]
            keypoints = keypoint_predictions[i]
            
            # Отправка данных через OSC
            # bbox_points = self.coord_transformer.create_bbox_points(x1, y1, x2, y2)
            # self.detector.send_bbox_data(bbox_points, i)
            
            # Отрисовка ключевых точек
            color = self.models.get_gesture_color(gesture_id)
            draw_image = self.models.draw_keypoints(draw_image, keypoints, bbox, color)
            # Получение мировых координат для кончика указательного пальца (точка 4)
            if len(keypoints) > 4:
                kp_x, kp_y = keypoints[4]
                abs_kp_x = x1 + int(kp_x * (x2 - x1) / self.models.crop_size)
                abs_kp_y = y1 + int(kp_y * (y2 - y1) / self.models.crop_size)
                
                # Получение мировых координат (можно добавить глубину если доступна)
                # Пример: depth = get_depth_from_tof_camera(abs_kp_x, abs_kp_y)
                # world_x, world_y = self.coord_transformer.get_world_coordinates_for_keypoint(
                #     abs_kp_x, abs_kp_y, depth
                # )
                
                # Текущий метод без глубины (гомография для Z=0)
                world_x, world_y = self.coord_transformer.get_worlcoordinates_for_keypoint(
                    abs_kp_x, abs_kp_y
                )
                
                # Отрисовка мировых координат
                self.detector.draw_world_coordinates(draw_image, world_x, world_y, color, i)
            
            # Отрисовка информации о детекции
            draw_image = self.detector.draw_detection_info(
                draw_image, bbox, gesture_id, track_id=0
            )
        
        # Сохранение отладочного изображения
        self.detector.save_debug_image(draw_image)
        
        return draw_image
    
    # def _display_frame(self, frame: np.ndarray):
    #     """Отображение кадра"""
    #     cv2.imshow("Hand Detection", frame)
    
    def _handle_keyboard_input(self) -> bool:
        """Обработка ввода с клавиатуры"""
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            return True
        return False
    
    def cleanup(self):
        """Очистка ресурсов"""
        print("Cleaning up...")
        self.running = False
        cv2.destroyAllWindows()
        self.camera.close()
        print("Cleanup completed")
    
    def stop(self):
        """Остановка приложения"""
        self.running = False
