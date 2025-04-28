import time
import matplotlib.pyplot as plt
from ultralytics import YOLO
import torch

# Load the YOLO Large model
model = YOLO('yolov8m.pt')

# Start timer
start_time = time.time()

print(torch.cuda.is_available())  # Should print True
print(torch.cuda.device_count())  # Should print the number of GPUs
print(torch.cuda.get_device_name(0))  # Should print your GPU name


# Train the model and store results
results = model.train(
    data='data.yaml', 
    epochs=75,  # Increased epochs
    imgsz=640,
    workers=0,
    batch=16,
    project='runs/train', 
    name='exp_medium',  # Changed directory name
    plots=True  # Enables built-in plotting
   
)

# End timer
end_time = time.time()
training_time = end_time - start_time
print(f"Training Time: {training_time:.2f} seconds")

# Extract training metrics
metrics = results.metrics  # Training metrics

# Plot and save train/val loss graph
plt.figure(figsize=(12, 5))
plt.plot(metrics['train/box_loss'], label='Train Box Loss', linestyle='--')
plt.plot(metrics['val/box_loss'], label='Val Box Loss')
plt.plot(metrics['train/cls_loss'], label='Train Cls Loss', linestyle='--')
plt.plot(metrics['val/cls_loss'], label='Val Cls Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.grid()
plt.savefig('runs/train/exp_medium/train_val_loss.png')  # Save plot
plt.show()

# Evaluate model on test set
test_results = model.val()

# Extract test metrics
test_metrics = test_results.metrics

# Plot and save test accuracy graph
plt.figure(figsize=(12, 5))
plt.bar(['mAP50', 'mAP50-95'], [test_metrics['metrics/mAP_50'], test_metrics['metrics/mAP_50-95']], color=['blue', 'green'])
plt.xlabel('Metric')
plt.ylabel('Value')
plt.title('Test Accuracy Metrics')
plt.savefig('runs/train/exp_large/test_accuracy.png')  # Save plot
plt.show()
