import torch
import torch.onnx
import timm




import tensorrt as trt
import numpy as np

import tensorrt as trt

def build_engine_onnx(onnx_file_path, engine_file_path, fp16_mode=True):
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(TRT_LOGGER)
    
    # Включаем explicit batch
    EXPLICIT_BATCH = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(EXPLICIT_BATCH)
    
    # Парсим ONNX
    parser = trt.OnnxParser(network, TRT_LOGGER)
    with open(onnx_file_path, 'rb') as f:
        if not parser.parse(f.read()):
            print("Ошибка парсинга ONNX")
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return None

    # Создаём config
    config = builder.create_builder_config()

    # === ИСПРАВЛЕНО: max_workspace_size → set_memory_pool_limit ===
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1 GB

    # Включаем FP16, если нужно
    if fp16_mode and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)

    # Добавляем optimization profile (если dynamic shape)
    profile = builder.create_optimization_profile()
    profile.set_shape('input', min=(1, 3, 256, 256), opt=(4, 3, 256, 256), max=(4, 3, 256, 256))
    config.add_optimization_profile(profile)

    # Строим engine
    try:
        serialized_engine = builder.build_serialized_network(network, config)
    except Exception as e:
        print("Ошибка при построении engine:", e)
        return None

    # Сохраняем engine
    with open(engine_file_path, "wb") as f:
        f.write(serialized_engine)
    
    print(f"TensorRT engine сохранён как {engine_file_path}")
    return serialized_engine

build_engine_onnx("my_model.onnx", "my_model.engine", fp16_mode=True)
