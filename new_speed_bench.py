from ultralytics import YOLO
import time
import numpy as np
import cv2
import torch
from collections import defaultdict
import timm
from torchvision import transforms
from PIL import Image
from run_trt import load_hand_gesture_model
# from train_kps import BlazeHandLandmark
# from run_trt import load_hand_gesture_model, predict_hand_gesture
import time
from PIL import Image
from polygraphy.backend.trt import EngineFromBytes, TrtRunner
import os
import time

INPUT_SIZE = (256, 256)

def run_model_once(image, runner: TrtRunner):
    """
    Делает один прого́н модели TensorRT через Polygraphy.
    Совместимо с polygraphy >= 1.0.0.
    """
    # Подготовка изображения
    if isinstance(image, np.ndarray):
        img = image
    elif isinstance(image, Image.Image):
        img = np.array(image)
    else:
        raise ValueError("Изображение должно быть PIL.Image или numpy.ndarray")

    # Убедимся, что RGB
    if img.ndim == 3 and img.shape[-1] == 3:
        pass  # предполагаем RGB
    else:
        raise ValueError("Изображение должно быть 3-канальным (RGB)")

    # Изменение размера
    img = cv2.resize(img, INPUT_SIZE, interpolation=cv2.INTER_LANCZOS4)  # (256, 256, 3)

    # HWC -> CHW, нормализация, float32
    img = img.transpose(2, 0, 1).astype(np.float32) / 255.0  # [3, 256, 256]

    # Добавляем batch: [1, 3, 256, 256]
    input_tensor = np.expand_dims(img, axis=0)

    # === ИСПОЛЬЗУЕМ КОНТЕКСТНЫЙ МЕНЕДЖЕР И ПОЛУЧАЕМ ИМЕНА ЧЕРЕЗ API ===
    with runner:
        # Получаем имя первого входа
        input_name = runner.engine[0]  # или runner.engine.inputs[0].name
        output_name = runner.engine[1]  # или runner.engine.outputs[0].name

        start_time = time.time()
        outputs = runner.infer({input_name: input_tensor})
        inference_time = time.time() - start_time

    # Получаем результат по имени выхода
    keypoints_3d = outputs[output_name]  # shape: [1, 21, 3]

    return keypoints_3d[0], inference_time  # возвращаем [21, 3]


def run_model_batch(images, runner: TrtRunner, image_size = 256, max_batch_size = 4):
    """
    Принимает список изображений (1, 2, ... N) и делает inference батчом.
    Автоматически разбивает большие batch на меньшие части для совместимости с TensorRT профилями.

    Параметры:
        images: список из PIL.Image или numpy.ndarray
        runner: TrtRunner
        image_size: размер изображения для ресайза
        max_batch_size: максимальный размер batch (по умолчанию 4 для совместимости с TensorRT профилями)

    Возвращает:
        keypoints_3d: np.ndarray [N, 21, 3]
        inference_time: float
    """
    if not images:
        return np.array([]), 0.0
    
    # Если количество изображений меньше или равно max_batch_size, обрабатываем как обычно
    if len(images) <= max_batch_size:
        batch = []
        for img in images:
            if isinstance(img, np.ndarray):
                pass
            elif isinstance(img, Image.Image):
                img = np.array(img)
            else:
                raise ValueError("Изображение должно быть PIL.Image или numpy.ndarray")

            img = cv2.resize(img, (image_size, image_size))
            img = img.transpose(2, 0, 1).astype(np.float32) / 255.0  # [3, H, W]
            batch.append(img)

        input_tensor = np.stack(batch)  # [N, 3, H, W]
        with runner:
            input_name = runner.engine[0]
            output_name = runner.engine[1]

            start_time = time.time()
            outputs = runner.infer({input_name: input_tensor})
            inference_time = time.time() - start_time
        if '794' in outputs:
            return outputs[output_name], outputs['794'], inference_time  # [N, 21, 3]
        return outputs[output_name], inference_time  # [N, 21, 3]
    
    # Если изображений больше max_batch_size, разбиваем на части
    all_outputs = []
    total_inference_time = 0.0
    
    for i in range(0, len(images), max_batch_size):
        batch_images = images[i:i + max_batch_size]
        
        batch = []
        for img in batch_images:
            if isinstance(img, np.ndarray):
                pass
            elif isinstance(img, Image.Image):
                img = np.array(img)
            else:
                raise ValueError("Изображение должно быть PIL.Image или numpy.ndarray")

            img = cv2.resize(img, (image_size, image_size))
            img = img.transpose(2, 0, 1).astype(np.float32) / 255.0  # [3, H, W]
            batch.append(img)

        input_tensor = np.stack(batch)  # [N, 3, H, W]

        with runner:
            input_name = runner.engine[0]
            output_name = runner.engine[1]

            start_time = time.time()
            outputs = runner.infer({input_name: input_tensor})
            batch_inference_time = time.time() - start_time
            total_inference_time += batch_inference_time

        all_outputs.append(outputs[output_name])
    
    # Объединяем результаты всех batch
    final_output = np.concatenate(all_outputs, axis=0)  # [N, 21, 3]
    
    return final_output, total_inference_time

if __name__ == 'main':
    model = load_hand_gesture_model("kps.engine")
    # model = YOLO("yolo11n.engine")

    # trt_model = load_hand_gesture_model("mobilenetv3_hand_cls.engine")
    image = cv2.imread('crops_Rest1007_3/Nikita_10.07/0_Nikita00108006.png')
    cropped_resized = cv2.resize(image, (256,256))
    for i in range(100):
        if i < 10:
            continue
        st = time.time()
        # results = model.predict(image)
        results = run_model_batch([image, image, image, image], model)
        
        # cls_class, conf = predict_hand_gesture(cropped_resized, trt_model)
        end = time.time()
        print(f'TIME {(end - st) * 1000}')