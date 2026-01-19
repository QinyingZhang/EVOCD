import os
from ultralytics import YOLO

# Load a pretrained segmentation model like YOLO11n-seg
model = YOLO("yolo11n.pt")  # load a pretrained model (recommended for training)

script_dir = os.path.dirname(os.path.abspath(__file__))
data_yaml_absolute_path = os.path.join(script_dir, "rear_side_d.yaml")

if __name__ == '__main__':
    results = model.train(
        lr0=0.01,
        data=data_yaml_absolute_path,
        epochs=300,
        imgsz=640,
        batch=8,
        project="./runs",
        name="train_n_01", 
        exist_ok=True,
        degrees=10.0,      
        translate=0.3,      
        scale=0.5,        
        flipud=0.5,         
        fliplr=0.5,        
        mosaic=0.8,        
        mixup=0.2,         
        copy_paste=0.2,    
        hsv_h=0.015,       
        hsv_s=0.7,          
        hsv_v=0.4,          
        patience=50,
        close_mosaic=100,   
        cls=1.0,            
    )