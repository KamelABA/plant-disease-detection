"""
Plant Disease Detection Model Training Script
Using Transfer Learning with EfficientNet and ResNet

This script trains improved models using:
- Transfer Learning from pre-trained ImageNet models
- Data Augmentation for better generalization
- Learning Rate Scheduling for optimal training
- Early Stopping to prevent overfitting

Usage:
    python train_model.py --crop potato --model efficientnet --epochs 50
    python train_model.py --crop tomato --model resnet50 --epochs 100

Author: PlantDetect Team
"""

import os
import argparse
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import (
    EfficientNetB0,
    EfficientNetB2,
    ResNet50,
    VGG16,
    MobileNetV2
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
    TensorBoard
)
from tensorflow.keras.optimizers import Adam
from datetime import datetime
import json

# ============================================================================
# CONFIGURATION
# ============================================================================

# Disease classes for each crop type
CROP_CLASSES = {
    'potato': ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy'],
    'tomato': [
        'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight',
        'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 
        'Tomato___Spider_mites_Two_spotted_spider_mite', 'Tomato___Target_Spot',
        'Tomato___Yellow_Leaf_Curl_Virus', 'Tomato___mosaic_virus', 'Tomato___healthy'
    ],
    'grape': [
        'Grape___Black_rot', 'Grape___Esca_(Black_Measles)',
        'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy'
    ],
    'corn': [
        'Corn_(maize)___Cercospora_leaf_spot_Gray_leaf_spot',
        'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight',
        'Corn_(maize)___healthy'
    ],
    'cotton': [
        'diseased cotton leaf', 'diseased cotton plant',
        'fresh cotton leaf', 'fresh cotton plant'
    ],
    'apple': [
        'Apple___Apple_scab', 'Apple___Black_rot',
        'Apple___Cedar_apple_rust', 'Apple___healthy'
    ]
}

# Image settings
IMG_SIZE = 224  # Standard for most pretrained models
BATCH_SIZE = 32

# ============================================================================
# DATA AUGMENTATION
# ============================================================================

def create_data_generators(train_dir, val_dir, img_size=IMG_SIZE, batch_size=BATCH_SIZE):
    """
    Create training and validation data generators with augmentation.
    
    Training augmentation includes:
    - Rotation, shift, shear, zoom
    - Horizontal/vertical flip
    - Brightness adjustment
    """
    
    # Training data with heavy augmentation
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        brightness_range=[0.8, 1.2],
        fill_mode='nearest'
    )
    
    # Validation data - only rescale, no augmentation
    val_datagen = ImageDataGenerator(rescale=1./255)
    
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True
    )
    
    val_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )
    
    return train_generator, val_generator

# ============================================================================
# MODEL ARCHITECTURES
# ============================================================================

def create_efficientnet_model(num_classes, img_size=IMG_SIZE, fine_tune_layers=20):
    """
    Create model using EfficientNetB0 as base.
    
    EfficientNet advantages:
    - Best accuracy-to-parameters ratio
    - Compound scaling (depth, width, resolution)
    - Lightweight yet powerful
    """
    
    # Load pre-trained EfficientNetB0
    base_model = EfficientNetB0(
        weights='imagenet',
        include_top=False,
        input_shape=(img_size, img_size, 3)
    )
    
    # Freeze base model initially
    base_model.trainable = False
    
    # Build the model
    inputs = keras.Input(shape=(img_size, img_size, 3))
    
    # Data augmentation layers (applied during training only)
    x = layers.RandomFlip("horizontal")(inputs)
    x = layers.RandomRotation(0.1)(x)
    x = layers.RandomZoom(0.1)(x)
    
    # Base model
    x = base_model(x, training=False)
    
    # Custom classification head
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = keras.Model(inputs, outputs)
    
    return model, base_model


def create_resnet_model(num_classes, img_size=IMG_SIZE):
    """
    Create model using ResNet50 as base.
    
    ResNet advantages:
    - Skip connections prevent vanishing gradients
    - Deep architecture captures complex features
    - Well-tested on plant datasets
    """
    
    base_model = ResNet50(
        weights='imagenet',
        include_top=False,
        input_shape=(img_size, img_size, 3)
    )
    
    base_model.trainable = False
    
    inputs = keras.Input(shape=(img_size, img_size, 3))
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = keras.Model(inputs, outputs)
    
    return model, base_model


def create_mobilenet_model(num_classes, img_size=IMG_SIZE):
    """
    Create model using MobileNetV2 as base.
    
    MobileNet advantages:
    - Extremely lightweight
    - Fast inference
    - Good for mobile/edge deployment
    """
    
    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=(img_size, img_size, 3)
    )
    
    base_model.trainable = False
    
    inputs = keras.Input(shape=(img_size, img_size, 3))
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = keras.Model(inputs, outputs)
    
    return model, base_model


def get_model(model_name, num_classes, img_size=IMG_SIZE):
    """Get model by name."""
    models = {
        'efficientnet': create_efficientnet_model,
        'resnet50': create_resnet_model,
        'mobilenet': create_mobilenet_model,
    }
    
    if model_name not in models:
        raise ValueError(f"Unknown model: {model_name}. Choose from: {list(models.keys())}")
    
    return models[model_name](num_classes, img_size)

# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

def get_callbacks(model_name, crop_name, patience=10):
    """
    Create training callbacks for:
    - Early stopping
    - Model checkpointing
    - Learning rate reduction
    - TensorBoard logging
    """
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = f"models/{crop_name}_{model_name}_{timestamp}.h5"
    log_dir = f"logs/{crop_name}_{model_name}_{timestamp}"
    
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    callbacks = [
        # Stop training when validation loss stops improving
        EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True,
            verbose=1
        ),
        
        # Save best model
        ModelCheckpoint(
            model_path,
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        
        # Reduce learning rate when plateau
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=5,
            min_lr=1e-7,
            verbose=1
        ),
        
        # TensorBoard logging
        TensorBoard(
            log_dir=log_dir,
            histogram_freq=1
        )
    ]
    
    return callbacks, model_path


def train_model(
    crop_name,
    model_name='efficientnet',
    train_dir='data/train',
    val_dir='data/val',
    epochs=50,
    initial_lr=1e-3,
    fine_tune_epochs=20,
    fine_tune_lr=1e-5
):
    """
    Train a plant disease detection model with transfer learning.
    
    Training happens in two phases:
    1. Train only the top layers (frozen base)
    2. Fine-tune the entire model with low learning rate
    """
    
    print("=" * 60)
    print(f"🌿 Training {crop_name.upper()} Disease Detection Model")
    print(f"📦 Base Model: {model_name}")
    print("=" * 60)
    
    # Get number of classes
    if crop_name in CROP_CLASSES:
        num_classes = len(CROP_CLASSES[crop_name])
    else:
        # Auto-detect from directory
        num_classes = len(os.listdir(train_dir))
    
    print(f"\n📊 Number of classes: {num_classes}")
    
    # Create data generators
    print("\n📁 Loading data...")
    train_gen, val_gen = create_data_generators(train_dir, val_dir)
    
    print(f"   Training samples: {train_gen.samples}")
    print(f"   Validation samples: {val_gen.samples}")
    
    # Create model
    print(f"\n🏗️ Building {model_name} model...")
    model, base_model = get_model(model_name, num_classes)
    
    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=initial_lr),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    model.summary()
    
    # Get callbacks
    callbacks, model_path = get_callbacks(model_name, crop_name)
    
    # ========================================
    # PHASE 1: Train top layers only
    # ========================================
    print("\n" + "=" * 60)
    print("📈 PHASE 1: Training top layers (base frozen)")
    print("=" * 60)
    
    history = model.fit(
        train_gen,
        epochs=epochs,
        validation_data=val_gen,
        callbacks=callbacks,
        verbose=1
    )
    
    # ========================================
    # PHASE 2: Fine-tune entire model
    # ========================================
    print("\n" + "=" * 60)
    print("📈 PHASE 2: Fine-tuning entire model")
    print("=" * 60)
    
    # Unfreeze the base model
    base_model.trainable = True
    
    # Freeze early layers, fine-tune later layers
    # (Early layers learn general features, later layers learn specific)
    for layer in base_model.layers[:-30]:
        layer.trainable = False
    
    # Recompile with lower learning rate
    model.compile(
        optimizer=Adam(learning_rate=fine_tune_lr),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Continue training
    fine_tune_history = model.fit(
        train_gen,
        epochs=fine_tune_epochs,
        validation_data=val_gen,
        callbacks=callbacks,
        verbose=1
    )
    
    # ========================================
    # Save final model
    # ========================================
    final_path = f"{crop_name.capitalize()}.h5"
    model.save(final_path)
    print(f"\n✅ Model saved to: {final_path}")
    
    # Save training history
    history_path = f"models/{crop_name}_{model_name}_history.json"
    with open(history_path, 'w') as f:
        json.dump({
            'phase1': {k: [float(v) for v in vals] for k, vals in history.history.items()},
            'phase2': {k: [float(v) for v in vals] for k, vals in fine_tune_history.history.items()}
        }, f)
    
    print(f"📊 Training history saved to: {history_path}")
    
    # Evaluate
    print("\n📊 Final Evaluation:")
    val_loss, val_acc = model.evaluate(val_gen)
    print(f"   Validation Loss: {val_loss:.4f}")
    print(f"   Validation Accuracy: {val_acc:.4f} ({val_acc*100:.2f}%)")
    
    return model, history


# ============================================================================
# QUICK TRAINING FUNCTIONS
# ============================================================================

def quick_train_potato():
    """Quick function to train potato model."""
    return train_model(
        crop_name='potato',
        model_name='efficientnet',
        train_dir='data/potato/train',
        val_dir='data/potato/val',
        epochs=30
    )


def quick_train_tomato():
    """Quick function to train tomato model."""
    return train_model(
        crop_name='tomato',
        model_name='efficientnet',
        train_dir='data/tomato/train',
        val_dir='data/tomato/val',
        epochs=50
    )


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Train Plant Disease Detection Model')
    parser.add_argument('--crop', type=str, required=True, 
                        help='Crop type (potato, tomato, grape, corn, cotton, apple)')
    parser.add_argument('--model', type=str, default='efficientnet',
                        choices=['efficientnet', 'resnet50', 'mobilenet'],
                        help='Base model architecture')
    parser.add_argument('--train_dir', type=str, default=None,
                        help='Training data directory')
    parser.add_argument('--val_dir', type=str, default=None,
                        help='Validation data directory')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--fine_tune_epochs', type=int, default=20,
                        help='Number of fine-tuning epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    
    args = parser.parse_args()
    
    # Set default directories if not provided
    train_dir = args.train_dir or f'data/{args.crop}/train'
    val_dir = args.val_dir or f'data/{args.crop}/val'
    
    # Check if data directories exist
    if not os.path.exists(train_dir):
        print(f"❌ Training directory not found: {train_dir}")
        print("\nℹ️ Please organize your data as follows:")
        print(f"   {train_dir}/")
        print(f"      ├── class1/")
        print(f"      │   ├── image1.jpg")
        print(f"      │   └── image2.jpg")
        print(f"      ├── class2/")
        print(f"      │   └── ...")
        return
    
    # Train model
    train_model(
        crop_name=args.crop,
        model_name=args.model,
        train_dir=train_dir,
        val_dir=val_dir,
        epochs=args.epochs,
        fine_tune_epochs=args.fine_tune_epochs
    )


if __name__ == "__main__":
    main()
