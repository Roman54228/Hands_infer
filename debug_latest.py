from hand_detection_app import HandDetectionApp
import cv2
import os
import glob
from typing import List, Optional
import numpy as np
import shutil

shutil.rmtree('roma_images')
os.makedirs('roma_images')

class DebugHandDetectionApp(HandDetectionApp):
    """Версия приложения для отладки с локальными изображениями"""
    
    def __init__(self, 
                 yolo_model_path: str,
                 hand_gesture_model_path: str,
                 kps_model_path: str,
                 images_folder: str = "debug_images",
                 output_dir: str = 'roma_images',
                 osc_ip: str = "10.0.0.101",
                 osc_port: int = 5055):
        
        # Инициализация родительского класса без камеры
        self.models = None
        self.detector = None
        self.coord_transformer = None
        self.output_dir = output_dir
        self.osc_ip = osc_ip
        self.osc_port = osc_port
        
        # Инициализация компонентов для отладки
        self._init_debug_components(
            yolo_model_path, 
            hand_gesture_model_path, 
            kps_model_path
        )
        
        # Настройка для работы с изображениями
        self.images_folder = images_folder
        self.image_paths = self._load_image_paths()
        self.current_image_index = 0
        self.running = False
        
        # Создание директории для сохранения изображений
        self._create_output_directory()
    
    def _init_debug_components(self, yolo_model_path: str, hand_gesture_model_path: str, kps_model_path: str):
        """Инициализация компонентов без камеры"""
        from model_manager import ModelManager
        from detection_processor import DetectionProcessor
        from coordinate_transformer import CoordinateTransformer
        
        self.models = ModelManager(
            yolo_model_path, 
            hand_gesture_model_path, 
            kps_model_path,
            use_parallel=False,
            parallel_workers=2
        )
        self.detector = DetectionProcessor(self.osc_ip, self.osc_port)
        self.coord_transformer = CoordinateTransformer()
    
    def _load_image_paths(self) -> List[str]:
        """Загрузка путей к изображениям из папки"""
        if not os.path.exists(self.images_folder):
            print(f"Creating debug images folder: {self.images_folder}")
            os.makedirs(self.images_folder, exist_ok=True)
            print(f"Please add images to {self.images_folder} folder and restart")
            return []
        
        # Поддерживаемые форматы изображений
        extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.tif']
        image_paths = []
        
        for ext in extensions:
            pattern = os.path.join(self.images_folder, ext)
            image_paths.extend(glob.glob(pattern))
            # Также ищем файлы в подпапках
            pattern = os.path.join(self.images_folder, '**', ext)
            image_paths.extend(glob.glob(pattern, recursive=True))
        
        image_paths.sort()  # Сортируем для предсказуемого порядка
        
        if not image_paths:
            print(f"No images found in {self.images_folder}")
            print("Supported formats: jpg, jpeg, png, bmp, tiff, tif")
        else:
            print(f"Found {len(image_paths)} images in {self.images_folder}")
        
        return image_paths
    
    def initialize(self) -> bool:
        """Инициализация для отладочного режима"""
        print("Initializing Debug Hand Detection Application...")
        
        if not self.image_paths:
            print("No images available for processing")
            return False
        
        print("Debug application initialized successfully!")
        return True
    
    def run(self):
        """Основной цикл для отладки с изображениями"""
        if not self.initialize():
            return
        
        self.running = True
        print("Starting debug hand detection...")
        print("Controls:")
        print("  'n' or 'space' - next image")
        print("  'p' - previous image")
        print("  'r' - restart from first image")
        print("  'q' - quit")
        print(f"Processing image {self.current_image_index + 1}/{len(self.image_paths)}")
        
        try:
            while self.running:
                # Загрузка текущего изображения
                current_image_path = self.image_paths[self.current_image_index]
                # current_image_path = 'krasota_pinch/frame0003.png' 
                frame = self._load_image(current_image_path)
                
                if frame is None:
                    print(f"Failed to load image: {current_image_path}")
                    self._next_image()
                    continue
                
                # Обработка изображения
                processed_frame = self._process_frame(frame)
                
                # Добавление информации о текущем изображении
                info_text = f"Image {self.current_image_index + 1}/{len(self.image_paths)}: {os.path.basename(current_image_path)}"
                cv2.putText(processed_frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Отображение результата
                cv2.imwrite(f'roma_images/{self.current_image_index}.jpg', processed_frame)
                self.current_image_index += 1
                # Обработка клавиш
                if self._handle_keyboard_input():
                    break
        
        except KeyboardInterrupt:
            print("\nDebug application interrupted by user")
        except Exception as e:
            print(f"Error in debug loop: {e}")
        finally:
            self.cleanup()
    
    def _load_image(self, image_path: str) -> Optional[np.ndarray]:
        """Загрузка изображения из файла"""
        try:
            frame = cv2.imread(image_path)
            frame = cv2.flip(frame, -1)
            if frame is None:
                print(f"Could not load image: {image_path}")
                return None
            return frame
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return None
    
    def _next_image(self):
        """Переход к следующему изображению"""
        self.current_image_index = (self.current_image_index + 1) % len(self.image_paths)
        print(f"Switched to image {self.current_image_index + 1}/{len(self.image_paths)}")
    
    def _previous_image(self):
        """Переход к предыдущему изображению"""
        self.current_image_index = (self.current_image_index - 1) % len(self.image_paths)
        print(f"Switched to image {self.current_image_index + 1}/{len(self.image_paths)}")
    
    def _restart_images(self):
        """Возврат к первому изображению"""
        self.current_image_index = 0
        print(f"Restarted from image {self.current_image_index + 1}/{len(self.image_paths)}")
    
    def _handle_keyboard_input(self) -> bool:
        """Обработка ввода с клавиатуры для отладочного режима"""
        key = cv2.waitKey(0) & 0xFF
        
        if key == ord('q'):
            return True
        elif key == ord('n') or key == ord(' '):  # next image
            self._next_image()
        elif key == ord('p'):  # previous image
            self._previous_image()
        elif key == ord('r'):  # restart
            self._restart_images()
        
        return False
    
    def cleanup(self):
        """Очистка ресурсов"""
        print("Cleaning up debug application...")
        self.running = False
        cv2.destroyAllWindows()
        print("Debug cleanup completed")


def main():
    """Главная функция отладочного приложения"""
    # Конфигурация путей к моделям
    yolo_model_path = "upside.onnx"
    hand_gesture_model_path = "best_cls_kps.engine"
    kps_model_path = "kps.engine"
    
    # Конфигурация OSC
    osc_ip = "10.0.0.101"
    osc_port = 5055
    
    # Папка с изображениями для отладки
    images_folder = "krasota_pinch"
    
    # Создание и запуск отладочного приложения
    app = DebugHandDetectionApp(
        yolo_model_path=yolo_model_path,
        hand_gesture_model_path=hand_gesture_model_path,
        kps_model_path=kps_model_path,
        images_folder=images_folder,
        osc_ip=osc_ip,
        osc_port=osc_port
    )
    
    try:
        app.run()
    except KeyboardInterrupt:
        print("\nDebug application stopped by user")
    except Exception as e:
        print(f"Debug application error: {e}")
    finally:
        app.cleanup()


if __name__ == "__main__":
    main()
