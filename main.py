#!/usr/bin/env python3
"""
Main Pipeline for Video Deepfake Detection - Model A
Author: Kishor-04
Date: 2025-01-06
"""

import argparse
import sys
from pathlib import Path

def run_preprocessing():
    """Run data preprocessing pipeline (frames + faces extraction)"""
    print("\n" + "="*70)
    print("STEP 1: DATA PREPROCESSING (Video → Frames → Faces)")
    print("="*70)
    
    # Extract frames
    from src.preprocessing.video_to_frames import VideoFrameExtractor
    print("\n1.1 Extracting frames from videos...")
    extractor = VideoFrameExtractor()
    extractor.run()
    
    # Extract faces
    from src.preprocessing.face_extraction import FaceExtractor
    print("\n1.2 Extracting faces from frames...")
    face_extractor = FaceExtractor()
    face_extractor.run()
    
    print("\n✅ Preprocessing completed!")

def run_dataset_preparation():
    """Run dataset preparation (randomize folders + split data)"""
    print("\n" + "="*70)
    print("STEP 2: DATASET PREPARATION (Randomize + Split)")
    print("="*70)
    
    from src.training.prepare_dataset import prepare_and_save_splits
    prepare_and_save_splits()
    
    print("\n✅ Dataset preparation completed!")

def run_training():
    """Run model training (uses pre-prepared dataset)"""
    print("\n" + "="*70)
    print("STEP 3: MODEL TRAINING")
    print("="*70)
    
    from src.training.fine_tune import main as train_main
    train_main()

def run_evaluation():
    """Run model evaluation"""
    print("\n" + "="*70)
    print("STEP 4: MODEL EVALUATION")
    print("="*70)
    
    from src.evaluation.evaluate import main as eval_main
    eval_main()

def run_inference(video_path=None, batch_dir=None):
    """Run inference on videos"""
    print("\n" + "="*70)
    print("STEP 5: INFERENCE")
    print("="*70)
    
    import yaml
    import torch
    from src.models.efficientnet_model import load_pretrained_efficientnet
    from src.inference.predict_video import VideoDeepfakePredictor
    
    config_path = 'config.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Load model
    checkpoint_path = Path(config['training']['checkpoint_dir']) / 'best_model.pth'
    
    if not checkpoint_path.exists():
        print(f"❌ Error: Model checkpoint not found at {checkpoint_path}")
        print("   Please train the model first using: python main.py --mode train")
        sys.exit(1)
    
    architecture = config['model']['architecture']
    
    if 'efficientnet' in architecture:
        model = load_pretrained_efficientnet(
            model_name=architecture,
            num_classes=config['model']['num_classes']
        )
    else:
        from src.models.xception_model import load_pretrained_xception
        model = load_pretrained_xception(num_classes=config['model']['num_classes'])
    
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Create predictor
    predictor = VideoDeepfakePredictor(model, config_path)
    
    # Run inference
    if video_path:
        predictor.predict_video(Path(video_path))
    elif batch_dir:
        predictor.predict_batch(batch_dir)
    else:
        print("❌ Error: Please specify --video or --batch")

def main():
    parser = argparse.ArgumentParser(
        description='Video Deepfake Detection Pipeline - Model A',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run complete pipeline
  python main.py --mode all
  
  # Run individual steps
  python main.py --mode preprocess          # Extract frames + faces
  python main.py --mode prepare_dataset     # Randomize folders + split data
  python main.py --mode train               # Train model only
  python main.py --mode evaluate            # Evaluate model
  
  # Inference
  python main.py --mode inference --video test.mp4
  python main.py --mode inference --batch test_videos/
  
  # Quick training (prepare + train)
  python main.py --mode quick_train
        """
    )
    
    parser.add_argument(
        '--mode',
        type=str,
        choices=['all', 'preprocess', 'prepare_dataset', 'train', 'evaluate', 'inference', 'quick_train'],
        default='all',
        help='Pipeline mode to run'
    )
    parser.add_argument('--video', type=str, help='Single video for inference')
    parser.add_argument('--batch', type=str, help='Directory of videos for batch inference')
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("🎯 VIDEO DEEPFAKE DETECTION - MODEL A")
    print("   Author: Kishor-04")
    print("   Date: 2025-01-06 05:55:07 UTC")
    print("="*70)
    
    try:
        if args.mode == 'all':
            # Full pipeline
            run_preprocessing()
            run_dataset_preparation()
            run_training()
            run_evaluation()
            
        elif args.mode == 'preprocess':
            # Only extract frames and faces
            run_preprocessing()
            
        elif args.mode == 'prepare_dataset':
            # Only randomize folders and split data
            run_dataset_preparation()
            
        elif args.mode == 'train':
            # Only train model (requires pre-prepared dataset)
            run_training()
            
        elif args.mode == 'evaluate':
            # Only evaluate model
            run_evaluation()
            
        elif args.mode == 'inference':
            # Only run inference
            if not args.video and not args.batch:
                print("\n❌ Error: --video or --batch required for inference mode")
                parser.print_help()
                sys.exit(1)
            run_inference(args.video, args.batch)
            
        elif args.mode == 'quick_train':
            # Prepare dataset + train (skip preprocessing)
            run_dataset_preparation()
            run_training()
        
        print("\n" + "="*70)
        print("✅ PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*70 + "\n")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Process interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()