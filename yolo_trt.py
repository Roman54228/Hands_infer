from ultralytics import YOLO

# Load a model
model = YOLO("upside.pt")  # load an official model

# Export the model
model.export(format="engine")