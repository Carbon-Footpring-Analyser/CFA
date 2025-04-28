# Carbon Footprint Analyser (CFA)

🌱 **Carbon Footprint Analyser** is a machine learning project that uses object detection (YOLO) to identify and classify objects that contribute to carbon emissions. The goal is to aid environmental analysis through automated image detection.

---

## Features

- ⚡ **YOLOv8**-based object detection.
- 📦 Pre-trained models (`yolov8m.pt`, `yolo11n.pt`) included.
- 🏷️ Custom dataset with labeled images.
- 📊 Training and inference scripts (`train_model.py`, `model_run.py`).

---

## Project Structure

```bash
CFA/
├── images/          # Image data (training and validation sets)
├── labels/          # Corresponding label files for images
├── runs/            # YOLO training and prediction outputs
├── data.yaml        # Dataset configuration file
├── model_run.py     # Script to run inference with trained models
├── train_model.py   # Script to train the YOLO model
├── yolo11n.pt       # Custom lightweight trained model
├── yolov8m.pt       # Pre-trained YOLOv8 model
```

---

## Setup Instructions

1. **Clone the Repository:**

```bash
git clone https://github.com/Carbon-Footpring-Analyser/CFA.git
cd CFA
```

2. **Install Dependencies:**

Ensure you have Python 3.8+ installed.

```bash
pip install ultralytics
```

3. **Train the Model:**

```bash
python train_model.py
```

4. **Run Inference:**

```bash
python model_run.py
```

---

## Requirements

- Python 3.8+
- [Ultralytics YOLO library](https://docs.ultralytics.com/)
- CUDA-enabled GPU recommended (for faster training/inference)

---

## Dataset

- Custom images and labels based on objects contributing to carbon emissions.
- Configuration handled through `data.yaml`.

---

## License

This project is under MIT License.
---

## Contributing

Contributions are welcome! Feel free to fork the repo and submit a pull request.

---

## Contact

For any inquiries or suggestions, feel free to open an issue.
