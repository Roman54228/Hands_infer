import cv2
import numpy as np
from PIL import Image
from polygraphy.backend.trt import EngineFromBytes, TrtRunner
import os
import cv2


ENGINE_PATH = "mobilenetv3_hand_cls.engine"
INPUT_SIZE = (400, 400)  # (width, height)


def load_hand_gesture_model(engine_path: str) -> TrtRunner:
    if not os.path.exists(engine_path):
        raise FileNotFoundError(f"Engine file not found: {engine_path}")

    with open(engine_path, "rb") as f:
        engine = EngineFromBytes(f.read())
    runner = TrtRunner(engine)
    print(f"Model loaded from {engine_path}")
    return runner



def predict_hand_gesture(image: np.ndarray, model: TrtRunner):
    if image.ndim == 3 and image.shape[2] == 3:
        if image[0, 0][0] > image[0, 0][2]:  # BGR?
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        raise ValueError("Input image must be HxWx3")

    img_pil = Image.fromarray(image)
    img_resized = img_pil.resize(INPUT_SIZE, Image.BILINEAR)
    img_np = np.array(img_resized, dtype=np.float32) / 255.0

  
    img_np = np.transpose(img_np, (2, 0, 1))  # (3, H, W)
    img_np = np.expand_dims(img_np, axis=0)   # (1, 3, H, W)

    with model:
        outputs = model.infer(feed_dict={"input": img_np})

    output = outputs["output"].squeeze()
    confidence = float(np.max(output))
    predicted_class = int(np.argmax(output))

    return predicted_class, confidence


if __name__ == '__main__':
    im = cv2.imread('path')
    im = cv2.resize(im, (640, 640))

    trt_model = load_hand_gesture_model("mobilenetv3_hand_cls.engine")
    cls_class, conf = cls, conf = predict_hand_gesture(im, trt_model)
    print(cls_class)