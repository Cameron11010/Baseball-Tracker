from ultralytics import YOLO


model = YOLO("best.pt")

# Export to CoreML (all classes included)
model.export(
    format="coreml",
    nms=True,        # built-in non-max suppression
    dynamic=False,   # fixed input size
    half=False       # use full precision (works well with Neural Engine)
)
