from ultralytics import YOLO
import torch


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Train completely from scratch
    model = YOLO("yolov8n.yaml")

    model.train(
        data="baseball.yaml",
        epochs=200,
        imgsz=960,
        batch=16,
        device=device,
        project="runs/baseball_yolo",
        name="from_scratch_v1",
        optimizer="AdamW",
        lr0=1e-3,
        patience=30,
        mosaic=1.0,
        degrees=10,
        translate=0.1,
        scale=0.5,
        shear=0.2,
        flipud=0.0,
        fliplr=0.5,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        close_mosaic=20,
    )


if __name__ == "__main__":
    main()
