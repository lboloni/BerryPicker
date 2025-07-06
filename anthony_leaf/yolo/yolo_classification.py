from ultralytics import YOLO
import os

# ===== Load pretrained YOLO11n model =====
model = YOLO("yolo11n-cls.pt")

results = model.train(data="leaf", epochs=100, imgsz=64)