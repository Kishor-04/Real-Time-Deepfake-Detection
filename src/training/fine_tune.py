"""
Fine-tuning script for deepfake detection model
Author: Kishor-04
Date: 2025-01-04
"""

import torch
import yaml
from pathlib import Path
import sys

def main():
    # Load config
    config_path = 'config.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print("\n" + "="*70)
    print("🎯 DEEPFAKE DETECTION - MODEL A FINE-TUNING")
    print("   Author: Kishor-04")
    print("   Date: 2025-01-04")
    print("="*70)
    
    # Create data loaders
    print("\n📊 Preparing data loaders...")
    from src.training.dataset import create_data_loaders
    train_loader, val_loader, test_loader = create_data_loaders(config_path)
    
    # Load model based on architecture in config
    print("\n🧠 Loading model...")
    architecture = config['model']['architecture']
    
    if architecture == 'xception':
        print(f"   Architecture: Xception")
        from src.models.xception_model import load_pretrained_xception
        model = load_pretrained_xception(
            weights_path=config['model'].get('pretrained_weights'),
            num_classes=config['model']['num_classes']
        )
    elif 'efficientnet' in architecture:
        print(f"   Architecture: {architecture}")
        from src.models.efficientnet_model import load_pretrained_efficientnet
        model = load_pretrained_efficientnet(
            model_name=architecture,
            weights_path=config['model'].get('pretrained_weights'),
            num_classes=config['model']['num_classes']
        )
    else:
        raise ValueError(f"Unknown architecture: {architecture}")
    
    # Create trainer
    from src.training.train import DeepfakeTrainer
    trainer = DeepfakeTrainer(model, train_loader, val_loader, config_path)
    
    # Start training
    trainer.train()
    
    print("\n🎉 Fine-tuning completed successfully!")
    print("   Next step: Run evaluation with 'python main.py --mode evaluate'\n")

if __name__ == "__main__":
    main()