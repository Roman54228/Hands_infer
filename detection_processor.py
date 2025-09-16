"""
Модуль для обработки детекций и трекинга
"""
import numpy as np
import cv2
from collections import defaultdict
from typing import List, Tuple, Dict, Optional
from GMHD_osc_sender import Sender


class DetectionProcessor:
    """Класс для обработки детекций и отправки данных через OSC"""
    
    def __init__(self, osc_ip: str = "10.0.0.101", osc_port: int = 5055):
        self.sender = Sender(ip=osc_ip, port=osc_port, logging_level="DEBUG")
        self.previous_detections = defaultdict(list)
        self.frame_counter = 0
        self.processing_interval = 8  # Обрабатываем каждый 8-й кадр
    
    def should_process_frame(self) -> bool:
        """Проверка, нужно ли обрабатывать текущий кадр"""
        self.frame_counter += 1
        return self.frame_counter % self.processing_interval == 0
    
    def smooth_detection(self, track_id: int, current_detection: List[int]) -> List[int]:
        """Сглаживание детекций для уменьшения дрожания"""
        # Добавляем текущую детекцию для этого трека
        self.previous_detections[track_id].append(current_detection)
        
        # Ограничиваем количество сохранённых детекций
        if len(self.previous_detections[track_id]) > 2:
            self.previous_detections[track_id].pop(0)
        
        # Сглаживание: берём среднее значение последних 2 детекций
        smoothed_box = np.mean(self.previous_detections[track_id], axis=0).astype(int)
        return smoothed_box.tolist()
    
    def send_bbox_data(self, bbox_points: Tuple[Tuple[float, float, float], ...], 
                      bbox_index: int):
        """Отправка данных bounding box через OSC"""
        for j, pt in enumerate(bbox_points):
            self.sender.send(
                address=f"/bboxes/bbox_{bbox_index}/point_{j}", 
                data=[pt[0], pt[1], pt[2]]
            )
    
    def draw_detection_info(self, image: np.ndarray, bbox: Tuple[int, int, int, int],
                          gesture_id: int, track_id: int = 0) -> np.ndarray:
        """Отрисовка информации о детекции на изображении"""
        x1, y1, x2, y2 = bbox
        
        # Получение цвета для жеста
        gesture_colors = {
            0: (255, 0, 0),      # 2_hands
            1: (255, 128, 0),   # fist
            2: (102, 255, 255), # normal_hand
            3: (153, 0, 153),   # pinch
            4: (178, 102, 55)   # pointer
        }
        color = gesture_colors.get(gesture_id, (255, 255, 255))
        
        # Отрисовка bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        
        # Отрисовка ID трека
        cv2.putText(image, f'id:{track_id}', (x1, y1 - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        # breakpoint()
        cv2.circle(image, (int(image.shape[1] // 2), int(image.shape[0] // 2)), 5, (255,0,0), -1, )
        
        return image
    
    def draw_world_coordinates(self, image: np.ndarray, world_x: float, world_y: float,
                             color: Tuple[int, int, int], index_frame):
        """Отрисовка мировых координат на изображении"""
        cv2.putText(image, f'{world_x:.1f} {world_y:.1f}', (50, 50 + index_frame * 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    def save_debug_image(self, image: np.ndarray, filename_prefix: str = "roma_images"):
        """Сохранение отладочного изображения"""
        try:
            cv2.imwrite(f'{filename_prefix}/{self.frame_counter}.png', image)
        except Exception as e:
            print(f"Error saving debug image: {e}")
    
    def reset_frame_counter(self):
        """Сброс счетчика кадров"""
        self.frame_counter = 0
