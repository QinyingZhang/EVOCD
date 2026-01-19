import os
from ultralytics import YOLO

model = YOLO("yolo11n.pt")
script_dir = os.path.dirname(os.path.abspath(__file__))
data_yaml_absolute_path = os.path.join(script_dir, "rear_d.yaml")

if __name__ == '__main__':
    results = model.train(
        data=data_yaml_absolute_path,
        epochs=300,
        imgsz=640,
        batch=8,
        device=0,
        optimizer='AdamW',
        lr0=0.01,
        lrf=0.1,
        cos_lr=True,
        warmup_epochs=5,
        weight_decay=0.0005,
        project="./runs",
        name="train_n_01",         exist_ok=True,
                mosaic=1.0,
        mixup=0.2,
        degrees=10.0,
        translate=0.2,
        scale=0.2,
        shear=4.0,
        perspective=0.0005,
        hsv_h=0.02,
        hsv_s=0.8,
        hsv_v=0.5,
        flipud=0.0,
        fliplr=0.5,
                close_mosaic=50,                 freeze=15,
                cache=True,
        workers=2,
        patience=50,     )