#!/usr/bin/env python3
"""
Script to download pre-trained deepfake detection weights
Author: Kishor-04
Date: 2025-01-04
"""

import os
import requests
from pathlib import Path
from tqdm import tqdm

def download_file(url, destination):
    """Download a file with progress bar"""
    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        destination.parent.mkdir(parents=True, exist_ok=True)
        
        with open(destination, 'wb') as file, tqdm(
            desc=destination.name,
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as progress_bar:
            for data in response.iter_content(chunk_size=1024):
                size = file.write(data)
                progress_bar.update(size)
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error downloading: {e}")
        return False

def download_pretrained_weights():
    """Download pre-trained weights for deepfake detection"""
    
    weights_dir = Path('models/pretrained')
    weights_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*70)
    print("📥 PRE-TRAINED WEIGHTS DOWNLOADER")
    print("="*70)
    
    # Available models
    models = {
        '1': {
            'name': 'EfficientNet-B0 (Light, Fast) ⭐ Recommended',
            'url': 'https://github.com/selimsef/dfdc_deepfake_challenge/releases/download/0.0.1/final_888_DeepFakeClassifier_tf_efficientnet_b0_ns_0_23',
            'filename': 'efficientnet_b0_deepfake.pth',
            'size': '~20MB'
        },
        '2': {
            'name': 'EfficientNet-B4 (Balanced)',
            'url': 'https://github.com/selimsef/dfdc_deepfake_challenge/releases/download/0.0.1/final_999_DeepFakeClassifier_tf_efficientnet_b4_ns_0_40',
            'filename': 'efficientnet_b4_deepfake.pth',
            'size': '~75MB'
        },
        '3': {
            'name': 'EfficientNet-B7 (Best Accuracy)',
            'url': 'https://github.com/selimsef/dfdc_deepfake_challenge/releases/download/0.0.1/final_111_DeepFakeClassifier_tf_efficientnet_b7_ns_0_36',
            'filename': 'efficientnet_b7_deepfake.pth',
            'size': '~260MB'
        }
    }
    
    print("\n📦 Available pre-trained models:")
    for key, model in models.items():
        print(f"  {key}. {model['name']} ({model['size']})")
    
    print("\n  4. Skip - Use ImageNet weights only (No download)")
    
    choice = input("\n👉 Select model to download (1-4): ").strip()
    
    if choice == '4':
        print("\n✓ Will use ImageNet pre-trained weights (built-in)")
        print("  No download needed. Ready to train!\n")
        return None
    
    if choice not in models:
        print("❌ Invalid choice")
        return None
    
    selected_model = models[choice]
    destination = weights_dir / selected_model['filename']
    
    if destination.exists():
        print(f"\n✓ {selected_model['name']} already downloaded")
        overwrite = input("  Overwrite? (y/n): ").strip().lower()
        if overwrite != 'y':
            return str(destination)
    
    print(f"\n📥 Downloading {selected_model['name']}...")
    print(f"   Size: {selected_model['size']}")
    
    success = download_file(selected_model['url'], destination)
    
    if success:
        print(f"\n✅ Downloaded successfully!")
        print(f"   Location: {destination}")
        
        print("\n" + "="*70)
        print("📝 NEXT STEPS:")
        print("="*70)
        print(f"\n1. Update your config.yaml:")
        print(f"\n   model:")
        print(f"     pretrained_weights: \"{destination}\"")
        print(f"\n2. Run training:")
        print(f"   python main.py --mode train")
        print("\n" + "="*70 + "\n")
        
        return str(destination)
    else:
        print("\n⚠️  Download failed. Will use ImageNet weights instead.")
        return None

if __name__ == "__main__":
    download_pretrained_weights()