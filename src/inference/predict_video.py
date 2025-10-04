import torch
import cv2
from pathlib import Path
import yaml
from tqdm import tqdm
import numpy as np
from facenet_pytorch import MTCNN
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import pandas as pd

class VideoDeepfakePredictor:
    """Video deepfake predictor"""
    
    def __init__(self, model, config_path='config.yaml'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.model = model
        self.device = torch.device(
            self.config['hardware']['device'] 
            if torch.cuda.is_available() else 'cpu'
        )
        self.model.to(self.device)
        self.model.eval()
        
        # Face detector
        print(f"Initializing MTCNN on {self.device}...")
        self.mtcnn = MTCNN(
            image_size=self.config['model']['input_size'][0],
            margin=40,
            device=self.device,
            post_process=False
        )
        
        # Transform
        self.transform = A.Compose([
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ])
        
        self.confidence_threshold = self.config['inference']['confidence_threshold']
    
    def extract_frames(self, video_path, max_frames=30):
        """Extract frames from video"""
        cap = cv2.VideoCapture(str(video_path))
        frames = []
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_interval = max(1, total_frames // max_frames)
        
        frame_count = 0
        while cap.isOpened() and len(frames) < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % frame_interval == 0:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
            
            frame_count += 1
        
        cap.release()
        return frames
    
    def predict_frame(self, frame):
        """Predict if a single frame is fake"""
        try:
            # Extract face
            img = Image.fromarray(frame)
            face = self.mtcnn(img)
            
            if face is None:
                # Use full frame if no face detected
                img_resized = cv2.resize(
                    frame,
                    tuple(self.config['model']['input_size'])
                )
                augmented = self.transform(image=img_resized)
                face_tensor = augmented['image'].unsqueeze(0)
            else:
                face_np = face.permute(1, 2, 0).numpy()
                face_np = (face_np * 255).astype(np.uint8)
                augmented = self.transform(image=face_np)
                face_tensor = augmented['image'].unsqueeze(0)
            
            # Predict
            with torch.no_grad():
                face_tensor = face_tensor.to(self.device)
                output = self.model(face_tensor)
                probability = torch.softmax(output, dim=1)[0]
                
            return probability[1].item()  # Probability of being fake
            
        except Exception as e:
            print(f"\n  ⚠️  Error processing frame: {e}")
            return None
    
    def predict_video(self, video_path):
        """Predict if a video is fake"""
        print(f"\n{'─'*70}")
        print(f"🎬 Analyzing: {video_path.name}")
        print(f"{'─'*70}")
        
        try:
            # Extract frames
            frames = self.extract_frames(video_path)
            print(f"  ✓ Extracted {len(frames)} frames")
            
            if not frames:
                print("  ❌ No frames extracted")
                return None, None
            
            # Predict for each frame
            predictions = []
            print(f"  🔍 Analyzing frames...")
            
            for frame in tqdm(frames, desc='  Progress', ncols=70):
                prob = self.predict_frame(frame)
                if prob is not None:
                    predictions.append(prob)
            
            if not predictions:
                print("  ❌ No valid predictions")
                return None, None
            
            # Aggregate predictions
            avg_probability = np.mean(predictions)
            std_probability = np.std(predictions)
            is_fake = avg_probability > self.confidence_threshold
            
            # Print results
            print(f"\n  {'='*66}")
            print(f"  📊 RESULTS:")
            print(f"  {'='*66}")
            print(f"  Fake Probability:  {avg_probability*100:>6.2f}%  (±{std_probability*100:.2f}%)")
            print(f"  Confidence:        {max(avg_probability, 1-avg_probability)*100:>6.2f}%")
            print(f"  Verdict:           {'🚨 FAKE' if is_fake else '✅ REAL':>10}")
            print(f"  {'='*66}")
            
            return is_fake, avg_probability
            
        except Exception as e:
            print(f"\n  ❌ Error: {str(e)}")
            return None, None
    
    def predict_batch(self, video_dir):
        """Predict for all videos in a directory"""
        video_dir = Path(video_dir)
        video_files = []
        
        for ext in ['*.mp4', '*.avi', '*.mov', '*.mkv', '*.MP4', '*.AVI', '*.MOV']:
            video_files.extend(video_dir.glob(ext))
        
        if not video_files:
            print(f"\n❌ No videos found in {video_dir}")
            return []
        
        print(f"\n{'='*70}")
        print(f"📁 BATCH PROCESSING: {len(video_files)} videos")
        print(f"{'='*70}")
        
        results = []
        fake_count = 0
        real_count = 0
        
        for video_path in video_files:
            is_fake, probability = self.predict_video(video_path)
            
            if is_fake is not None:
                results.append({
                    'video': video_path.name,
                    'is_fake': 'FAKE' if is_fake else 'REAL',
                    'fake_probability': f"{probability*100:.2f}%",
                    'confidence': f"{max(probability, 1-probability)*100:.2f}%"
                })
                
                if is_fake:
                    fake_count += 1
                else:
                    real_count += 1
        
        # Save results
        output_dir = Path(self.config['inference']['output_path'])
        output_dir.mkdir(exist_ok=True, parents=True)
        
        df = pd.DataFrame(results)
        csv_path = output_dir / 'predictions.csv'
        df.to_csv(csv_path, index=False)
        
        # Print summary
        print(f"\n{'='*70}")
        print(f"📊 BATCH SUMMARY")
        print(f"{'='*70}")
        print(f"  Total videos:  {len(results)}")
        print(f"  Real:          {real_count} ({real_count/len(results)*100:.1f}%)")
        print(f"  Fake:          {fake_count} ({fake_count/len(results)*100:.1f}%)")
        print(f"  Results saved: {csv_path}")
        print(f"{'='*70}\n")
        
        return results

def main():
    """Main inference function"""
    import argparse
    import yaml
    from src.models.efficientnet_model import load_pretrained_efficientnet
    from src.models.xception_model import load_pretrained_xception
    
    parser = argparse.ArgumentParser(description='Predict deepfake videos')
    parser.add_argument('--video', type=str, help='Path to a single video')
    parser.add_argument('--batch', type=str, help='Path to directory of videos')
    args = parser.parse_args()
    
    if not args.video and not args.batch:
        print("❌ Error: Please specify --video or --batch")
        parser.print_help()
        return
    
    config_path = 'config.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Load model
    checkpoint_path = Path(config['training']['checkpoint_dir']) / 'best_model.pth'
    
    if not checkpoint_path.exists():
        print(f"❌ Error: Model not found at {checkpoint_path}")
        print("   Please train the model first")
        return
    
    architecture = config['model']['architecture']
    
    if 'efficientnet' in architecture:
        model = load_pretrained_efficientnet(
            model_name=architecture,
            num_classes=config['model']['num_classes']
        )
    else:
        model = load_pretrained_xception(num_classes=config['model']['num_classes'])
    
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Create predictor
    predictor = VideoDeepfakePredictor(model, config_path)
    
    # Predict
    if args.video:
        predictor.predict_video(Path(args.video))
    elif args.batch:
        predictor.predict_batch(args.batch)

if __name__ == "__main__":
    main()