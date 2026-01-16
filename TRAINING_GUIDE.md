# 🧠 Model Training Guide

This guide explains how to train improved plant disease detection models using **Transfer Learning**.

## 📋 Prerequisites

1. **Python 3.8+**
2. **TensorFlow 2.x**
3. **GPU (recommended)** - Training is much faster with CUDA-enabled GPU

Install dependencies:
```bash
pip install tensorflow opencv-python numpy pillow
```

For GPU support:
```bash
pip install tensorflow[and-cuda]
```

## 📁 Data Organization

Organize your training data in the following structure:

```
data/
└── potato/
    ├── train/
    │   ├── Potato___Early_blight/
    │   │   ├── image001.jpg
    │   │   ├── image002.jpg
    │   │   └── ...
    │   ├── Potato___Late_blight/
    │   │   └── ...
    │   └── Potato___healthy/
    │       └── ...
    └── val/
        ├── Potato___Early_blight/
        ├── Potato___Late_blight/
        └── Potato___healthy/
```

### Getting Training Data

**Option 1: PlantVillage Dataset (Recommended)**
- Download from [Kaggle](https://www.kaggle.com/datasets/emmarex/plantdisease)
- Contains 54,000+ images across 38 classes
- High quality, well-labeled data

**Option 2: Your Own Data**
- Collect at least 100 images per class
- Include variety (lighting, angles, backgrounds)
- Balance classes as much as possible

### Splitting Data

Use the preparation script to split your data:
```bash
python prepare_data.py --action split --source data/raw/potato --output data/potato --split 0.2
```

This creates an 80/20 train/validation split.

## 🚀 Training

### Quick Start

Train a potato disease classifier:
```bash
python train_model.py --crop potato --model efficientnet --epochs 50
```

### Training Options

| Argument | Description | Default |
|----------|-------------|---------|
| `--crop` | Crop type (potato, tomato, grape, corn, cotton, apple) | Required |
| `--model` | Base model (efficientnet, resnet50, mobilenet) | efficientnet |
| `--epochs` | Training epochs | 50 |
| `--fine_tune_epochs` | Fine-tuning epochs | 20 |
| `--batch_size` | Batch size | 32 |
| `--train_dir` | Custom training directory | data/{crop}/train |
| `--val_dir` | Custom validation directory | data/{crop}/val |

### Model Comparison

| Model | Accuracy | Size | Speed | Best For |
|-------|----------|------|-------|----------|
| **EfficientNetB0** | ⭐⭐⭐⭐⭐ | 29 MB | Medium | Best overall |
| **ResNet50** | ⭐⭐⭐⭐ | 98 MB | Slow | High accuracy |
| **MobileNetV2** | ⭐⭐⭐ | 14 MB | Fast | Mobile/Edge |

### Training Process

The script uses a **two-phase training approach**:

**Phase 1: Feature Extraction (Frozen Base)**
- Base model layers are frozen
- Only the classification head is trained
- Uses higher learning rate (1e-3)
- Prevents destroying pre-trained features

**Phase 2: Fine-Tuning**
- Unfreezes last 30 layers of base model
- Uses very low learning rate (1e-5)
- Allows model to adapt to plant features
- Improves accuracy significantly

## 📊 Monitoring Training

### TensorBoard

Training logs are saved to the `logs/` folder. View them with:
```bash
tensorboard --logdir logs
```

Open http://localhost:6006 in your browser to see:
- Training/validation loss curves
- Accuracy metrics
- Learning rate changes

### Training Output

```
📈 PHASE 1: Training top layers (base frozen)
Epoch 1/50
100/100 [==============================] - 45s - loss: 0.8234 - accuracy: 0.7123 - val_loss: 0.4521 - val_accuracy: 0.8456

...

📈 PHASE 2: Fine-tuning entire model
Epoch 1/20
100/100 [==============================] - 65s - loss: 0.1234 - accuracy: 0.9623 - val_loss: 0.1021 - val_accuracy: 0.9756

✅ Model saved to: Potato.h5
📊 Final Accuracy: 97.56%
```

## 💡 Tips for Better Results

### 1. More Data is Better
- Aim for 1000+ images per class
- Use data augmentation (built into the script)
- Consider class balancing

### 2. Class Balancing
If you have imbalanced classes:
```python
# In train_model.py, add class weights
from sklearn.utils.class_weight import compute_class_weight
class_weights = compute_class_weight('balanced', ...)
model.fit(..., class_weight=class_weights)
```

### 3. Hyperparameter Tuning
Try different configurations:
```bash
# Higher epochs for more data
python train_model.py --crop potato --epochs 100 --fine_tune_epochs 30

# Smaller batch for limited GPU memory
python train_model.py --crop potato --batch_size 16
```

### 4. Data Augmentation
The script includes these augmentations:
- Random rotation (±40°)
- Random shift (20%)
- Random zoom (20%)
- Horizontal & vertical flip
- Brightness adjustment

### 5. Early Stopping
Training automatically stops when validation loss stops improving for 10 epochs, preventing overfitting.

## 🔄 Updating the Application

After training, the new model is saved as `{Crop}.h5` in the project root. The application will automatically use it on the next restart:

```bash
# Stop the current server (Ctrl+C)
# Start with new model
python app2.py
```

## 🐛 Troubleshooting

### Out of Memory (OOM)
- Reduce batch size: `--batch_size 16` or `--batch_size 8`
- Use MobileNet instead of ResNet

### Low Accuracy
1. Check class distribution: `python prepare_data.py --action analyze --source data/potato`
2. Add more training data
3. Increase epochs
4. Try a different base model

### Overfitting (val_loss increasing)
- Add more data
- Increase dropout (edit train_model.py)
- Reduce fine_tune_epochs

## 📈 Expected Results

With the PlantVillage dataset and EfficientNet:

| Crop | Classes | Expected Accuracy |
|------|---------|-------------------|
| Potato | 3 | 97-99% |
| Tomato | 10 | 94-97% |
| Grape | 4 | 96-98% |
| Corn | 4 | 95-97% |
| Apple | 4 | 96-98% |

---

**Happy Training! 🌱**
