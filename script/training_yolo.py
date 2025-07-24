from ultralytics import YOLO

# Load a YOLOv8 pretrained model (e.g., YOLOv8n)
model = YOLO("yolov8n.pt")  # Available variants: yolov8n.pt, yolov8s.pt, yolov8m.pt, etc.

# Train the model on your custom dataset
model.train(
    data="C:/Users/kumar/OneDrive/Desktop/Mini_project/data/data.yaml",  # Path to your YAML file
    epochs=50,  # Number of training epochs
    imgsz=640,  # Image size for training
    project="C:/Users/kumar/OneDrive/Desktop/Mini_project/runs",  # Custom project folder
    name="yolov8_training"  # Name of the training run
)
