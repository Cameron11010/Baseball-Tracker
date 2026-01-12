from ultralytics import YOLO


model = YOLO("best.pt")


model.export(
    format="coreml",
    nms=True,        
    dynamic=False,  
    half=False       
)
