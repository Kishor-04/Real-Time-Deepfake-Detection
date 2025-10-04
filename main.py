#!/usr/bin/env python3
"""
Main Pipeline for Video Deepfake Detection - Model A
Author: Kishor-04
Date: 2025-10-04
"""

import argparse
import sys
from pathlib import Path

def run_preprocessing():
    """Run data preprocessing pipeline"""
    print("\n" + "="*70)
    print("STEP 1: DATA PREPROCESSING")
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

def run_training():
    """Run model training"""
    print("\n" + "="*70)
    print("STEP 2: MODEL TRAINING")
    print("="*70)
    
    from src.training.fine_tune import main as train_main
    train_main()

def run_evaluation():
    """Run model evaluation"""
    print("\n" + "="*70)
    print("STEP 3: MODEL EVALUATION")
    print("="*70)
    
    from src.evaluation.evaluate import main as eval_main
    eval_main()

def run_inference(video_path=None, batch_dir=None):
    """Run inference on videos"""
    print("\n" + "="*70)
    print("STEP 4: INFERENCE")
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
    
    checkpoint = torch.load(checkpoint_path)
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
  python main.py --mode preprocess
  python main.py --mode train
  python main.py --mode evaluate
  
  # Inference
  python main.py --mode inference --video test.mp4
  python main.py --mode inference --batch test_videos/
        """
    )
    
    parser.add_argument(
        '--mode',
        type=str,
        choices=['all', 'preprocess', 'train', 'evaluate', 'inference'],
        default='all',
        help='Pipeline mode to run'
    )
    parser.add_argument('--video', type=str, help='Single video for inference')
    parser.add_argument('--batch', type=str, help='Directory of videos for batch inference')
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("🎯 VIDEO DEEPFAKE DETECTION - MODEL A")
    print("   Author: Kishor-04")
    print("   Date: 2025-10-04")
    print("="*70)
    
    try:
        if args.mode == 'all':
            run_preprocessing()
            run_training()
            run_evaluation()
        elif args.mode == 'preprocess':
            run_preprocessing()
        elif args.mode == 'train':
            run_training()
        elif args.mode == 'evaluate':
            run_evaluation()
        elif args.mode == 'inference':
            if not args.video and not args.batch:
                print("\n❌ Error: --video or --batch required for inference mode")
                parser.print_help()
                sys.exit(1)
            run_inference(args.video, args.batch)
        
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