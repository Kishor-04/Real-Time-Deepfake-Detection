"""
Dataset and DataLoader creation for deepfake detection
Author: Kishor-04
Date: 2025-01-06
UPDATED: Loads from train/val/test folder structure
"""

import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import cv2
import numpy as np
import yaml
import pickle

class DeepfakeDataset(Dataset):
    """Dataset class for deepfake detection"""
    
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Convert string path to Path object if needed
        if isinstance(img_path, str):
            img_path = Path(img_path)
        
        # Load image
        image = cv2.imread(str(img_path))
        
        if image is None:
            print(f"Warning: Could not load {img_path}")
            image = np.zeros((224, 224, 3), dtype=np.uint8)
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply transforms
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        
        return image, label

def load_prepared_splits(config_path='config.yaml'):
    """
    Load pre-prepared train/val/test splits from disk
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    splits_file = Path(config['dataset']['processed_data_path']) / 'splits' / 'data_splits.pkl'
    
    if not splits_file.exists():
        print(f"\n❌ Error: Pre-prepared splits not found at: {splits_file}")
        print(f"\n💡 Please run dataset preparation first:")
        print(f"   python main.py --mode prepare_dataset")
        raise FileNotFoundError(f"Splits file not found: {splits_file}")
    
    print(f"\n📂 Loading pre-prepared splits from: {splits_file}")
    
    with open(splits_file, 'rb') as f:
        splits = pickle.load(f)
    
    # Convert paths back to Path objects
    for split_name in ['train', 'val', 'test']:
        paths, labels = splits[split_name]
        splits[split_name] = ([Path(p) for p in paths], labels)
    
    print(f"  ✓ Loaded train split: {len(splits['train'][0]):,} frames")
    print(f"  ✓ Loaded val split:   {len(splits['val'][0]):,} frames")
    print(f"  ✓ Loaded test split:  {len(splits['test'][0]):,} frames")
    
    return splits

def create_data_loaders(config_path='config.yaml'):
    """Create DataLoaders for training from pre-prepared splits"""
    import sys
    sys.path.append('.')
    from src.preprocessing.data_augmentation import get_train_transforms, get_val_transforms
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Load pre-prepared splits
    data_splits = load_prepared_splits(config_path)
    
    # Create datasets
    train_dataset = DeepfakeDataset(
        data_splits['train'][0],
        data_splits['train'][1],
        transform=get_train_transforms(config)
    )
    
    val_dataset = DeepfakeDataset(
        data_splits['val'][0],
        data_splits['val'][1],
        transform=get_val_transforms()
    )
    
    test_dataset = DeepfakeDataset(
        data_splits['test'][0],
        data_splits['test'][1],
        transform=get_val_transforms()
    )
    
    # Create data loaders
    batch_size = config['training']['batch_size']
    num_workers = config['hardware']['num_workers']
    pin_memory = config['hardware']['pin_memory']
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    print(f"\n✅ DataLoaders created successfully")
    print(f"  Batch size: {batch_size}")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    print(f"  Test batches: {len(test_loader)}")
    
    return train_loader, val_loader, test_loader