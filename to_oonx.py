import timm
import torch
from torchvision import transforms

# --- Конфиг ---
num_classes = 5
ckpt_path = "hand_gesture_classifier.pth"
onnx_path = "my_model.onnx"
dummy_hw = (400, 400)  # Можно сменить на 384/416 и т.п., если нужно

# --- Модель ---
model_hands_cls = timm.create_model("fastvit_sa24", pretrained=True, num_classes=num_classes)
# альтернативно:
# model_hands_cls = timm.create_model("mobilenetv3_small_100", pretrained=True, num_classes=num_classes)

# Трансформы (для инференса в проде; для экспорта не обязательны)
transform_hands_cls = transforms.Compose([
    transforms.Resize(dummy_hw),
    transforms.ToTensor(),
    # Рекомендуется нормализация под ImageNet, если веса обучались на ней:
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# --- Загрузка весов (безопасно) ---
def load_weights_safely(model: torch.nn.Module, path: str):
    ckpt = torch.load(path, map_location="cpu")
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        ckpt = ckpt["state_dict"]
    if isinstance(ckpt, dict):
        # снимаем возможный префикс "module."
        ckpt = {k.replace("module.", ""): v for k, v in ckpt.items()}
        missing, unexpected = model.load_state_dict(ckpt, strict=False)
        if missing or unexpected:
            print(f"[WARN] missing={len(missing)}, unexpected={len(unexpected)}")
    else:
        # если вдруг в ckpt лежит не state_dict (маловероятно)
        model.load_state_dict(ckpt)

load_weights_safely(model_hands_cls, ckpt_path)

model_hands_cls.eval()
model_hands_cls.to("cpu")

# --- Манекен-вход ---
x = torch.randn(1, 3, dummy_hw[0], dummy_hw[1], dtype=torch.float32)

with torch.no_grad():
    torch.onnx.export(
        model_hands_cls,
        x,
        onnx_path,
        export_params=True,
        opset_version=17,                 # можно попробовать 16 или 18
        do_constant_folding=True,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={
            "input": {0: "batch", 2: "height", 3: "width"},
            "logits": {0: "batch"}
        }
    )