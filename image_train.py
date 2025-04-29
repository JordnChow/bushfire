from ultralytics import YOLO

model = YOLO('yolov8m-cls.pt')
model.train(data="./dataset", epochs=10, imgzs=224)