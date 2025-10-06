"""
Comprehensive diagnostic script to identify why Val Acc > Train Acc
"""

import torch
import torch.nn as nn
from pathlib import Path
import yaml
from collections import defaultdict
import numpy as np
from PIL import Image
import sys
sys.path.append('.')

from src.training.dataset import DeepfakeDataset
from src.models.efficientnet_model import EfficientNetDeepfakeDetector


def test_1_check_augmentation():
    """Test 1: Check if augmentation is causing the issue"""
    print("\n" + "=" * 80)
    print("TEST 1: AUGMENTATION ANALYSIS")
    print("=" * 80)
    
    # Load datasets
    train_dataset = DeepfakeDataset('data/processed/faces/train', split='train')
    val_dataset = DeepfakeDataset('data/processed/faces/val', split='val')
    
    print(f"\n📊 Dataset sizes:")
    print(f"   Train: {len(train_dataset)} images")
    print(f"   Val:   {len(val_dataset)} images")
    
    # Check transforms
    print(f"\n🎨 Augmentation check:")
    print(f"   Train transforms: {train_dataset.transform}")
    print(f"   Val transforms:   {val_dataset.transform}")
    
    # Sample comparison
    print(f"\n🔍 Loading sample images...")
    train_sample, _ = train_dataset[0]
    val_sample, _ = val_dataset[0]
    
    print(f"   Train sample shape: {train_sample.shape}")
    print(f"   Val sample shape:   {val_sample.shape}")
    
    if train_dataset.transform != val_dataset.transform:
        print("\n⚠️  FINDING: Different augmentation between train and val")
        print("   This explains ~1-2% difference, but NOT 4%+ difference")
    else:
        print("\n✅ Same transforms for train and val")


def test_2_model_behavior():
    """Test 2: Check model behavior in train vs eval mode"""
    print("\n" + "=" * 80)
    print("TEST 2: MODEL BEHAVIOR ANALYSIS")
    print("=" * 80)
    
    # Load model
    try:
        model = EfficientNetDeepfakeDetector()
        checkpoint = torch.load('models/checkpoints/best_model.pth', map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Create dummy input
        dummy_input = torch.randn(8, 3, 224, 224)
        
        # Test in training mode
        model.train()
        with torch.no_grad():
            train_mode_output = model(dummy_input)
            train_mode_preds = torch.softmax(train_mode_output, dim=1)
        
        # Test in eval mode
        model.eval()
        with torch.no_grad():
            eval_mode_output = model(dummy_input)
            eval_mode_preds = torch.softmax(eval_mode_output, dim=1)
        
        # Compare
        diff = torch.abs(train_mode_preds - eval_mode_preds).mean().item()
        
        print(f"\n📊 Model output difference (train vs eval mode):")
        print(f"   Mean absolute difference: {diff:.6f}")
        
        if diff > 0.05:
            print(f"\n⚠️  FINDING: Large difference between train/eval modes")
            print(f"   This suggests BatchNorm/Dropout causing variance")
            print(f"   Expected: <0.05, Actual: {diff:.6f}")
        else:
            print(f"\n✅ Normal difference between modes (< 0.05)")
            
    except Exception as e:
        print(f"\n❌ Could not load model: {e}")


def test_3_data_distribution():
    """Test 3: Check if validation set is easier/smaller"""
    print("\n" + "=" * 80)
    print("TEST 3: DATA DISTRIBUTION ANALYSIS")
    print("=" * 80)
    
    faces_dir = Path('data/processed/faces')
    
    stats = {}
    for split in ['train', 'val', 'test']:
        split_path = faces_dir / split
        if split_path.exists():
            real_files = list((split_path / 'real').rglob('*.jpg'))
            fake_files = list((split_path / 'fake').rglob('*.jpg'))
            
            # Extract video IDs
            real_videos = set()
            fake_videos = set()
            
            for f in real_files:
                # Get parent folder name as video ID
                video_id = f.parent.name if f.parent.name != 'real' else f.stem.split('_')[0]
                real_videos.add(video_id)
            
            for f in fake_files:
                video_id = f.parent.name if f.parent.name != 'fake' else f.stem.split('_')[0]
                fake_videos.add(video_id)
            
            stats[split] = {
                'real_images': len(real_files),
                'fake_images': len(fake_files),
                'real_videos': len(real_videos),
                'fake_videos': len(fake_videos),
                'frames_per_video': (len(real_files) + len(fake_files)) / (len(real_videos) + len(fake_videos)) if (len(real_videos) + len(fake_videos)) > 0 else 0
            }
    
    print(f"\n📊 Split Statistics:")
    print("-" * 80)
    print(f"{'Split':<8} {'Images':<12} {'Videos':<12} {'Frames/Video':<15} {'% of Total':<10}")
    print("-" * 80)
    
    total_images = sum(s['real_images'] + s['fake_images'] for s in stats.values())
    
    for split, data in stats.items():
        total_split_images = data['real_images'] + data['fake_images']
        total_split_videos = data['real_videos'] + data['fake_videos']
        percentage = (total_split_images / total_images * 100) if total_images > 0 else 0
        
        print(f"{split:<8} {total_split_images:<12,} {total_split_videos:<12} "
              f"{data['frames_per_video']:<15.1f} {percentage:<10.2f}%")
    
    # Check if validation is too small
    if 'val' in stats:
        val_percentage = (stats['val']['real_images'] + stats['val']['fake_images']) / total_images * 100
        val_videos = stats['val']['real_videos'] + stats['val']['fake_videos']
        
        print(f"\n🔍 Validation Set Analysis:")
        if val_percentage < 10:
            print(f"   ⚠️  WARNING: Validation set is only {val_percentage:.1f}% of data")
            print(f"   Recommended: 15-20%")
        
        if val_videos < 50:
            print(f"   ⚠️  WARNING: Only {val_videos} videos in validation")
            print(f"   This might not be representative!")
            print(f"   Recommended: >100 videos for reliable validation")
        
        if val_percentage >= 15 and val_videos >= 50:
            print(f"   ✅ Validation set size looks reasonable")


def test_4_check_video_leakage_detailed():
    """Test 4: Detailed video overlap check"""
    print("\n" + "=" * 80)
    print("TEST 4: DETAILED VIDEO LEAKAGE CHECK")
    print("=" * 80)
    
    faces_dir = Path('data/processed/faces')
    
    # Collect all video IDs with their file counts
    split_videos = defaultdict(lambda: defaultdict(lambda: {'videos': set(), 'frame_counts': defaultdict(int)}))
    
    for split in ['train', 'val', 'test']:
        split_path = faces_dir / split
        if not split_path.exists():
            continue
        
        for class_type in ['real', 'fake']:
            class_path = split_path / class_type
            if not class_path.exists():
                continue
            
            for img_file in class_path.rglob('*.jpg'):
                # Extract video ID (parent folder name)
                video_id = img_file.parent.name
                if video_id == class_type:
                    # Flat structure, extract from filename
                    video_id = img_file.stem.split('_')[0]
                
                split_videos[split][class_type]['videos'].add(video_id)
                split_videos[split][class_type]['frame_counts'][video_id] += 1
    
    # Check for overlaps
    print(f"\n🔍 Checking for video overlaps...")
    
    found_leakage = False
    splits = list(split_videos.keys())
    
    for i, split1 in enumerate(splits):
        for split2 in splits[i+1:]:
            for class_type in ['real', 'fake']:
                videos1 = split_videos[split1][class_type]['videos']
                videos2 = split_videos[split2][class_type]['videos']
                overlap = videos1 & videos2
                
                if overlap:
                    found_leakage = True
                    print(f"\n🚨 LEAKAGE FOUND!")
                    print(f"   {split1.upper()} ↔ {split2.upper()} ({class_type}): {len(overlap)} videos overlap")
                    print(f"   Sample overlapping videos: {list(overlap)[:5]}")
    
    if found_leakage:
        print(f"\n❌ CRITICAL: Video leakage detected!")
        print(f"   This is THE PRIMARY CAUSE of val_acc > train_acc")
        return False
    else:
        print(f"\n✅ No video leakage detected")
        return True


def test_5_sample_predictions():
    """Test 5: Compare model predictions on train vs val samples"""
    print("\n" + "=" * 80)
    print("TEST 5: SAMPLE PREDICTION ANALYSIS")
    print("=" * 80)
    
    try:
        # Load model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = EfficientNetDeepfakeDetector()
        checkpoint = torch.load('models/checkpoints/best_model.pth', map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        
        # Load datasets
        train_dataset = DeepfakeDataset('data/processed/faces/train', split='train')
        val_dataset = DeepfakeDataset('data/processed/faces/val', split='val')
        
        # Sample predictions
        num_samples = 100
        train_correct = 0
        val_correct = 0
        
        print(f"\n🔍 Testing {num_samples} random samples from each split...")
        
        with torch.no_grad():
            # Test training samples
            for _ in range(num_samples):
                idx = np.random.randint(0, len(train_dataset))
                img, label = train_dataset[idx]
                img = img.unsqueeze(0).to(device)
                
                output = model(img)
                pred = output.argmax(dim=1).item()
                if pred == label:
                    train_correct += 1
            
            # Test validation samples
            for _ in range(num_samples):
                idx = np.random.randint(0, len(val_dataset))
                img, label = val_dataset[idx]
                img = img.unsqueeze(0).to(device)
                
                output = model(img)
                pred = output.argmax(dim=1).item()
                if pred == label:
                    val_correct += 1
        
        train_acc = train_correct / num_samples * 100
        val_acc = val_correct / num_samples * 100
        
        print(f"\n📊 Spot Check Results:")
        print(f"   Train accuracy: {train_acc:.2f}%")
        print(f"   Val accuracy:   {val_acc:.2f}%")
        print(f"   Difference:     {val_acc - train_acc:+.2f}%")
        
        if val_acc > train_acc + 2:
            print(f"\n⚠️  FINDING: Val accuracy significantly higher")
            print(f"   This confirms the issue exists with current model/data")
        else:
            print(f"\n✅ Reasonable accuracy relationship")
            
    except Exception as e:
        print(f"\n❌ Could not perform prediction test: {e}")


def main():
    """Run all diagnostic tests"""
    print("\n" + "🔬 " * 20)
    print(" " * 20 + "COMPREHENSIVE DIAGNOSTIC SUITE")
    print(" " * 18 + "Val Acc > Train Acc Investigation")
    print("🔬 " * 20)
    
    results = {}
    
    # Run all tests
    try:
        test_1_check_augmentation()
        results['augmentation'] = True
    except Exception as e:
        print(f"\n❌ Test 1 failed: {e}")
        results['augmentation'] = False
    
    try:
        test_2_model_behavior()
        results['model_behavior'] = True
    except Exception as e:
        print(f"\n❌ Test 2 failed: {e}")
        results['model_behavior'] = False
    
    try:
        test_3_data_distribution()
        results['data_distribution'] = True
    except Exception as e:
        print(f"\n❌ Test 3 failed: {e}")
        results['data_distribution'] = False
    
    try:
        no_leakage = test_4_check_video_leakage_detailed()
        results['no_leakage'] = no_leakage
    except Exception as e:
        print(f"\n❌ Test 4 failed: {e}")
        results['no_leakage'] = False
    
    try:
        test_5_sample_predictions()
        results['predictions'] = True
    except Exception as e:
        print(f"\n❌ Test 5 failed: {e}")
        results['predictions'] = False
    
    # Final diagnosis
    print("\n" + "=" * 80)
    print("🎯 FINAL DIAGNOSIS")
    print("=" * 80)
    
    if not results.get('no_leakage', True):
        print("\n🚨 PRIMARY ISSUE: DATA LEAKAGE")
        print("   → Same videos appear in both train and validation splits")
        print("   → This is causing unrealistically high validation accuracy")
        print("\n💡 SOLUTION:")
        print("   1. Stop training immediately")
        print("   2. Re-run preprocessing with video-level splitting")
        print("   3. Ensure each video appears in ONLY ONE split")
        print("   4. Restart training from scratch")
    else:
        print("\n🔍 POSSIBLE CAUSES:")
        print("   • Augmentation making training harder (minor effect)")
        print("   • Small/non-representative validation set")
        print("   • Dropout/BatchNorm behavior differences")
        print("\n💡 RECOMMENDATIONS:")
        print("   1. Check if validation set is representative")
        print("   2. Ensure sufficient validation samples (>100 videos)")
        print("   3. Monitor training for a few more epochs")
        print("   4. Gap should decrease as training progresses")
    
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
