from ultralytics import YOLO
model = YOLO("./train4/weights/best.pt")
results_fire = model("C:/Users/jorda/Downloads/8e5462bfc5e48d9038160a2766007b29ebb4df80-1920x1080.jpg")