# to_onnx_to_trt.py

import torch
import torch.onnx
import numpy as np
import os
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit  # <-- ОБЯЗАТЕЛЬНО: инициализация CUDA контекста


IMAGE_SIZE = 256
# ==============================================================================
# 3. Калибратор для INT8
# ==============================================================================

class Int8Calibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, calibration_dataset, batch_size, cache_file):
        super().__init__()
        self.calibration_dataset = calibration_dataset  # список: [array(3,256,256), ...]
        self.batch_size = batch_size
        self.current_index = 0
        self.cache_file = cache_file
        self.device_inputs = None
        self.data_shape = (3, IMAGE_SIZE, IMAGE_SIZE)

    def get_batch_size(self):
        return self.batch_size

    def get_batch(self, names):
        if self.current_index >= len(self.calibration_dataset):
            return None

        # ЛЕНИВАЯ ИНИЦИАЛИЗАЦИЯ: выделяем память только при первом вызове
        if self.device_inputs is None:
            self.device_inputs = cuda.mem_alloc(
                trt.volume(self.data_shape) * self.batch_size * np.float32().nbytes
            )

        # Собираем батч
        batch = []
        for i in range(self.batch_size):
            if self.current_index >= len(self.calibration_dataset):
                break
            img = self.calibration_dataset[self.current_index]
            batch.append(img)
            self.current_index += 1

        if not batch:
            return None

        # На CPU: [B, 3, 256, 256]
        batch = np.ascontiguousarray(np.stack(batch)).astype(np.float32)

        # Копируем на GPU
        cuda.memcpy_htod(self.device_inputs, batch)

        # Возвращаем список указателей (по одному на вход)
        return [self.device_inputs]

    def read_calibration_cache(self):
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                return f.read()
        return None

    def write_calibration_cache(self, cache):
        with open(self.cache_file, "wb") as f:
            f.write(cache)


# ==============================================================================
# 4. Построение TensorRT Engine
# ==============================================================================

def build_engine_onnx(
    onnx_file_path,
    engine_file_path,
    fp16_mode=True,
    int8_mode=False,
    calib_dataset=None,
    calib_batch_size=4,
    calib_cache="calibration_cache.bin"
):
    logger = trt.Logger(trt.Logger.INFO)  # INFO чтобы видеть процесс
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    config = builder.create_builder_config()

    # Парсим ONNX
    parser = trt.OnnxParser(network, logger)
    with open(onnx_file_path, "rb") as f:
        if not parser.parse(f.read()):
            print("❌ Ошибка парсинга ONNX:")
            for e in range(parser.num_errors):
                print(parser.get_error(e))
            return None

    # Память
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1 GB

    # FP16
    if fp16_mode and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        print("✅ FP16 включён")

    # INT8
    if int8_mode:
        if not builder.platform_has_fast_int8:
            print("⚠️ Устройство не поддерживает INT8")
        else:
            config.set_flag(trt.BuilderFlag.INT8)
            print("✅ INT8 включён")

            if calib_dataset is None:
                raise ValueError("INT8 требует calib_dataset")

            calibrator = Int8Calibrator(calib_dataset, calib_batch_size, calib_cache)
            #config.set_int8_calibrator(calibrator)
            config.int8_calibrator = calibrator

    # Optimization profile
    profile = builder.create_optimization_profile()
    profile.set_shape("input", min=(1, 3, IMAGE_SIZE, IMAGE_SIZE), opt=(2, 3, IMAGE_SIZE, IMAGE_SIZE), max=(4, 3, IMAGE_SIZE, IMAGE_SIZE))
    config.add_optimization_profile(profile)

    # Строим engine
    print("🔨 Строим TensorRT engine... (может занять 1–5 минут при INT8)")
    try:
        serialized_engine = builder.build_serialized_network(network, config)
    except Exception as e:
        print("❌ Ошибка при сборке engine:", e)
        return None

    if serialized_engine is None:
        print("❌ Не удалось построить engine")
        return None

    # Сохраняем
    with open(engine_file_path, "wb") as f:
        f.write(serialized_engine)
    print(f"✅ TensorRT engine сохранён: {engine_file_path}")
    return serialized_engine


# ==============================================================================
# 5. Генерация калибровочных данных (пример)
# ==============================================================================

def create_calibration_dataset(num_samples=200):
    print("📊 Генерация калибровочных данных...")
    dataset = []
    for _ in range(num_samples):
        # Заменить на реальные изображения!
        img = np.random.uniform(0, 1, (3, IMAGE_SIZE, IMAGE_SIZE)).astype(np.float32)  # нормализованные [0,1]
        dataset.append(img)
    return dataset


# ==============================================================================
# 6. Запуск всего пайплайна
# ==============================================================================

if __name__ == "__main__":
    # Шаг 1: ONNX

    # Шаг 2: Калибровочные данные
    calib_dataset = create_calibration_dataset(num_samples=200)  # Замени на реальные изображения!
    onnx_path = 'cls_kps.onnx'
    # Шаг 3: Сборка engine
    build_engine_onnx(
        onnx_file_path=onnx_path,
        engine_file_path="cls_kps.engine",
        fp16_mode=True,
        int8_mode=False,             # <-- включаем INT8
        calib_dataset=calib_dataset,
        calib_batch_size=1,
        calib_cache="int8_calibration.cache"
    )

    print("🎉 Готово!")