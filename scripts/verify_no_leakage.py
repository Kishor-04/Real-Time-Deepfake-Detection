
import os
from pathlib import Path
from collections import defaultdict
import json

def extract_video_id(file_path):
    """
    Extract video ID from file path or filename
    Supports multiple naming conventions
    """
    path = Path(file_path)
    filename = path.stem  # Filename without extension
    
    # Method 1: Check if parent folder is a video ID (numeric or video name)
    parent_folder = path.parent.name
    if parent_folder not in ['real', 'fake', 'train', 'val', 'test']:
        return parent_folder
    
    # Method 2: Extract from filename patterns
    # Pattern: videoID_frame_001.jpg or video_001_frame_005.jpg
    parts = filename.split('_')
    if len(parts) >= 2:
        # First part is usually video ID
        return parts[0]
    
    return filename


def check_video_leakage():
    """Check if same video appears in multiple splits (train/val/test)"""
    
    print("\n" + "=" * 80)
    print("🔍 CHECKING FOR VIDEO LEAKAGE BETWEEN SPLITS")
    print("=" * 80)
    
    faces_dir = Path('data/processed/faces')
    
    if not faces_dir.exists():
        print("❌ ERROR: Faces directory not found!")
        return False, []
    
    # Collect video IDs from each split
    split_videos = defaultdict(lambda: {'real': set(), 'fake': set()})
    
    for split in ['train', 'val', 'test']:
        split_path = faces_dir / split
        if not split_path.exists():
            continue
        
        # Process real videos
        real_path = split_path / 'real'
        if real_path.exists():
            for img_file in list(real_path.rglob('*.jpg')) + list(real_path.rglob('*.png')):
                video_id = extract_video_id(img_file)
                split_videos[split]['real'].add(video_id)
        
        # Process fake videos
        fake_path = split_path / 'fake'
        if fake_path.exists():
            for img_file in list(fake_path.rglob('*.jpg')) + list(fake_path.rglob('*.png')):
                video_id = extract_video_id(img_file)
                split_videos[split]['fake'].add(video_id)
    
    # Print video counts per split
    print(f"\n📊 UNIQUE VIDEOS PER SPLIT:")
    print("-" * 80)
    for split in ['train', 'val', 'test']:
        if split in split_videos:
            real_count = len(split_videos[split]['real'])
            fake_count = len(split_videos[split]['fake'])
            total_count = real_count + fake_count
            print(f"  {split.upper():8} - Real: {real_count:4} videos | Fake: {fake_count:4} videos | "
                  f"Total: {total_count:4} videos")
    print("-" * 80)
    
    # Check for overlaps
    leakage_found = False
    overlaps = []
    
    splits = list(split_videos.keys())
    for i, split1 in enumerate(splits):
        for split2 in splits[i+1:]:
            for class_type in ['real', 'fake']:
                overlap = split_videos[split1][class_type] & split_videos[split2][class_type]
                if overlap:
                    leakage_found = True
                    overlaps.append({
                        'split1': split1,
                        'split2': split2,
                        'class': class_type,
                        'videos': overlap
                    })
    
    # Report findings
    print(f"\n🔎 LEAKAGE CHECK RESULTS:")
    print("-" * 80)
    
    if leakage_found:
        print("🚨 DATA LEAKAGE DETECTED! 🚨\n")
        for overlap_info in overlaps:
            print(f"  ⚠️  {overlap_info['split1'].upper()} ↔ {overlap_info['split2'].upper()} "
                  f"({overlap_info['class']}): {len(overlap_info['videos'])} overlapping videos")
            
            # Show first 10 overlapping videos
            sample_videos = list(overlap_info['videos'])[:10]
            for vid in sample_videos:
                print(f"      - Video: {vid}")
            if len(overlap_info['videos']) > 10:
                print(f"      ... and {len(overlap_info['videos']) - 10} more")
            print()
        
        print("❌ CRITICAL: Same videos appear in multiple splits!")
        print("   This causes the model to perform unrealistically well on validation.")
        print("   You must re-split your data at the VIDEO level, not FRAME level.")
    else:
        print("✅ NO LEAKAGE DETECTED!")
        print("   Each video appears in only one split (train/val/test)")
        print("   Data split is properly isolated.")
    
    print("-" * 80)
    
    return not leakage_found, overlaps

def check_folder_structure():
    """Check if data is properly organized into train/val folders"""
    
    faces_dir = Path('data/processed/faces')
    
    print("\n" + "=" * 80)
    print("📂 CHECKING FOLDER STRUCTURE")
    print("=" * 80)
    
    if not faces_dir.exists():
        print("❌ ERROR: Faces directory not found!")
        return False, {}
    
    # Check if train/val/test folders exist
    has_train_folder = (faces_dir / 'train').exists()
    has_val_folder = (faces_dir / 'val').exists()
    has_test_folder = (faces_dir / 'test').exists()
    
    print(f"\n{'✅' if has_train_folder else '❌'} Train folder exists: {has_train_folder}")
    print(f"{'✅' if has_val_folder else '❌'} Val folder exists: {has_val_folder}")
    print(f"{'✅' if has_test_folder else '❌'} Test folder exists: {has_test_folder}")
    
    if not has_train_folder or not has_val_folder:
        print("\n🚨 CRITICAL: Missing train/val folder structure!")
        print("   This means data is being split at FRAME level → LEAKAGE RISK!")
        print("\n💡 Correct structure should be:")
        print("   data/processed/faces/")
        print("   ├── train/")
        print("   │   ├── real/")
        print("   │   └── fake/")
        print("   ├── val/")
        print("   │   ├── real/")
        print("   │   └── fake/")
        print("   └── test/")
        print("       ├── real/")
        print("       └── fake/")
        return False, {}
    
    # Count files in each split
    stats = {}
    for split in ['train', 'val', 'test']:
        split_path = faces_dir / split
        if split_path.exists():
            real_files = list((split_path / 'real').rglob('*.jpg')) + list((split_path / 'real').rglob('*.png'))
            fake_files = list((split_path / 'fake').rglob('*.jpg')) + list((split_path / 'fake').rglob('*.png'))
            stats[split] = {
                'real': len(real_files),
                'fake': len(fake_files),
                'total': len(real_files) + len(fake_files)
            }
    
    print(f"\n📊 DATA DISTRIBUTION:")
    print("-" * 80)
    total_all = sum(s['total'] for s in stats.values())
    for split, data in stats.items():
        percentage = (data['total'] / total_all * 100) if total_all > 0 else 0
        print(f"  {split.upper():8} - Real: {data['real']:6,} | Fake: {data['fake']:6,} | "
              f"Total: {data['total']:6,} ({percentage:5.2f}%)")
    print(f"  {'TOTAL':8} - {total_all:,} images")
    print("-" * 80)
    
    return True, stats


def check_mapping_files():
    """Check split mapping files for consistency"""
    
    print("\n" + "=" * 80)
    print("� CHECKING SPLIT MAPPING FILES")
    print("=" * 80)
    
    processed_dir = Path('data/processed')
    mapping_files = [
        'faces/real_split_mapping.txt',
        'faces/fake_split_mapping.txt',
        'splits/video_split_mapping.txt'
    ]
    
    all_exist = True
    for mapping_file in mapping_files:
        file_path = processed_dir / mapping_file
        exists = file_path.exists()
        print(f"  {'✅' if exists else '❌'} {mapping_file}: {'Found' if exists else 'Missing'}")
        if not exists:
            all_exist = False
    
    if all_exist:
        print("\n✅ All mapping files present")
        print("   You can verify splits by checking these files.")
    else:
        print("\n⚠️  Some mapping files are missing")
        print("   This might indicate incomplete preprocessing.")
    
    print("-" * 80)
    
    return all_exist


if __name__ == '__main__':
    print("\n" + "🔍 " * 20)
    print(" " * 25 + "DATA LEAKAGE VERIFICATION")
    print("🔍 " * 20)
    
    # Step 1: Check folder structure
    has_proper_structure, stats = check_folder_structure()
    
    # Step 2: Check for video leakage
    no_leakage, overlaps = check_video_leakage()
    
    # Step 3: Check mapping files
    has_mappings = check_mapping_files()
    
    # Final summary
    print("\n" + "=" * 80)
    print("📋 FINAL VERDICT")
    print("=" * 80)
    
    all_checks_passed = has_proper_structure and no_leakage
    
    if all_checks_passed:
        print("\n✅ ✅ ✅  ALL CHECKS PASSED!  ✅ ✅ ✅")
        print("\n  Your data split appears to be correct:")
        print("  • Proper folder structure ✓")
        print("  • No video leakage between splits ✓")
        print("  • Training can proceed safely ✓")
        print("\n  Expected behavior:")
        print("  • Train accuracy should be ≥ Validation accuracy")
        print("  • Small gap (1-3%) is healthy and normal")
    else:
        print("\n🚨 🚨 🚨  ISSUES DETECTED!  🚨 🚨 🚨")
        print("\n  Problems found:")
        if not has_proper_structure:
            print("  ❌ Improper folder structure")
        if not no_leakage:
            print("  ❌ Video leakage between splits")
        print("\n  ⚠️  DO NOT CONTINUE TRAINING!")
        print("  Your model's high validation accuracy is likely due to data leakage.")
        print("\n  Recommended actions:")
        print("  1. Stop current training")
        print("  2. Re-run preprocessing with proper video-level split")
        print("  3. Verify no leakage again")
        print("  4. Restart training from scratch")
    
    print("=" * 80 + "\n")
    
    # Exit with appropriate code
    exit(0 if all_checks_passed else 1)