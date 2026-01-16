"""
Data Preparation Utility for Plant Disease Detection

This script helps prepare and organize your training data.
It can:
1. Split data into train/validation sets
2. Analyze class distribution
3. Download sample datasets

Usage:
    python prepare_data.py --crop potato --split 0.2
"""

import os
import shutil
import random
import argparse
from collections import Counter

def split_dataset(source_dir, output_dir, val_split=0.2, seed=42):
    """
    Split dataset into training and validation sets.
    
    Args:
        source_dir: Directory containing class folders with images
        output_dir: Output directory for train/val split
        val_split: Fraction of data for validation (default 0.2 = 20%)
        seed: Random seed for reproducibility
    """
    random.seed(seed)
    
    train_dir = os.path.join(output_dir, 'train')
    val_dir = os.path.join(output_dir, 'val')
    
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    
    print(f"📁 Source: {source_dir}")
    print(f"📁 Output: {output_dir}")
    print(f"📊 Validation split: {val_split*100:.0f}%")
    print()
    
    total_train = 0
    total_val = 0
    
    # Get all class directories
    classes = [d for d in os.listdir(source_dir) 
               if os.path.isdir(os.path.join(source_dir, d))]
    
    for class_name in classes:
        class_path = os.path.join(source_dir, class_name)
        
        # Get all images in class
        images = [f for f in os.listdir(class_path) 
                  if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        
        if not images:
            print(f"⚠️ No images found in: {class_name}")
            continue
        
        # Shuffle and split
        random.shuffle(images)
        split_idx = int(len(images) * (1 - val_split))
        train_images = images[:split_idx]
        val_images = images[split_idx:]
        
        # Create class directories
        train_class_dir = os.path.join(train_dir, class_name)
        val_class_dir = os.path.join(val_dir, class_name)
        os.makedirs(train_class_dir, exist_ok=True)
        os.makedirs(val_class_dir, exist_ok=True)
        
        # Copy images
        for img in train_images:
            shutil.copy2(
                os.path.join(class_path, img),
                os.path.join(train_class_dir, img)
            )
        
        for img in val_images:
            shutil.copy2(
                os.path.join(class_path, img),
                os.path.join(val_class_dir, img)
            )
        
        total_train += len(train_images)
        total_val += len(val_images)
        
        print(f"✅ {class_name}: {len(train_images)} train, {len(val_images)} val")
    
    print()
    print(f"📊 Total: {total_train} training, {total_val} validation images")
    return train_dir, val_dir


def analyze_dataset(data_dir):
    """
    Analyze class distribution in a dataset.
    """
    print(f"\n📊 Dataset Analysis: {data_dir}")
    print("=" * 50)
    
    if not os.path.exists(data_dir):
        print(f"❌ Directory not found: {data_dir}")
        return
    
    class_counts = {}
    total = 0
    
    for class_name in sorted(os.listdir(data_dir)):
        class_path = os.path.join(data_dir, class_name)
        if os.path.isdir(class_path):
            images = [f for f in os.listdir(class_path) 
                     if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
            count = len(images)
            class_counts[class_name] = count
            total += count
    
    # Print analysis
    for class_name, count in sorted(class_counts.items(), key=lambda x: -x[1]):
        pct = count / total * 100 if total > 0 else 0
        bar = "█" * int(pct / 2)
        print(f"{class_name[:40]:40s} | {count:5d} ({pct:5.1f}%) {bar}")
    
    print("-" * 50)
    print(f"{'Total':40s} | {total:5d}")
    print(f"\n📈 Classes: {len(class_counts)}")
    
    # Check for imbalance
    if class_counts:
        min_count = min(class_counts.values())
        max_count = max(class_counts.values())
        if max_count > min_count * 3:
            print("⚠️ Warning: Dataset is imbalanced!")
            print("   Consider using class weights or oversampling.")
    
    return class_counts


def download_plantvillage_sample():
    """
    Instructions for downloading PlantVillage dataset.
    """
    print("""
╔══════════════════════════════════════════════════════════════╗
║              📥 PlantVillage Dataset Download                 ║
╠══════════════════════════════════════════════════════════════╣
║                                                               ║
║  The PlantVillage dataset contains 54,000+ images of         ║
║  healthy and diseased plant leaves across 38 classes.        ║
║                                                               ║
║  Download options:                                            ║
║                                                               ║
║  1. Kaggle (Recommended):                                     ║
║     https://www.kaggle.com/datasets/emmarex/plantdisease     ║
║                                                               ║
║  2. GitHub (Original):                                        ║
║     https://github.com/spMohanty/PlantVillage-Dataset        ║
║                                                               ║
║  3. Direct (Segmented):                                       ║
║     https://data.mendeley.com/datasets/tywbtsjrjv/1          ║
║                                                               ║
║  After downloading, organize as:                              ║
║                                                               ║
║  data/                                                        ║
║  └── plantvillage/                                           ║
║      ├── Potato___Early_blight/                              ║
║      │   ├── image1.jpg                                      ║
║      │   └── ...                                             ║
║      ├── Potato___Late_blight/                               ║
║      └── ...                                                 ║
║                                                               ║
║  Then run:                                                    ║
║  python prepare_data.py --action split --source data/plantvillage ║
║                                                               ║
╚══════════════════════════════════════════════════════════════╝
""")


def main():
    parser = argparse.ArgumentParser(description='Prepare training data')
    parser.add_argument('--action', type=str, default='analyze',
                        choices=['split', 'analyze', 'download'],
                        help='Action to perform')
    parser.add_argument('--source', type=str, default='data/raw',
                        help='Source directory with class folders')
    parser.add_argument('--output', type=str, default='data',
                        help='Output directory')
    parser.add_argument('--split', type=float, default=0.2,
                        help='Validation split ratio')
    parser.add_argument('--crop', type=str, default=None,
                        help='Crop name for organization')
    
    args = parser.parse_args()
    
    if args.action == 'download':
        download_plantvillage_sample()
    
    elif args.action == 'split':
        output = args.output
        if args.crop:
            output = os.path.join(args.output, args.crop)
        split_dataset(args.source, output, args.split)
    
    elif args.action == 'analyze':
        if os.path.exists(os.path.join(args.source, 'train')):
            analyze_dataset(os.path.join(args.source, 'train'))
            analyze_dataset(os.path.join(args.source, 'val'))
        else:
            analyze_dataset(args.source)


if __name__ == "__main__":
    main()
