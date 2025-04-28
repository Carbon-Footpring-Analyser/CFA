from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO('runs/train/exp_medium6/weights/best.pt')
    model.val(data='data.yaml')

    model.predict(source='0', show=True, conf=0.5)  # Webcam; adjust source for video files
