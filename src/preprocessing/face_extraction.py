import cv2
import os
from pathlib import Path
import yaml
from tqdm import tqdm
from facenet_pytorch import MTCNN
import torch
from PIL import Image
import numpy as np

class FaceExtractor:
    """Extract faces from frames using MTCNN"""
    
    def __init__(self, config_path='config.yaml'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Initialize MTCNN for face detection
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Initializing MTCNN on {device}...")
        
        self.mtcnn = MTCNN(
            image_size=self.config['model']['input_size'][0],
            margin=40,
            device=device,
            post_process=False
        )
        
    def extract_face(self, image_path):
        """Extract face from an image"""
        try:
            img = Image.open(image_path).convert('RGB')
            
            # Detect and extract face
            face = self.mtcnn(img)
            
            if face is not None:
                return face
            else:
                # If no face detected, return resized original image
                img_array = np.array(img)
                img_resized = cv2.resize(
                    img_array, 
                    tuple(self.config['model']['input_size'])
                )
                return torch.from_numpy(img_resized).permute(2, 0, 1)
                
        except Exception as e:
            print(f"\n  ⚠️  Error extracting face from {image_path}: {str(e)}")
            return None
    
    def process_frames(self, frames_dir, output_dir):
        """Process all frames and extract faces"""
        frames_dir = Path(frames_dir)
        output_dir = Path(output_dir)
        
        for label in ['real', 'fake']:
            label_dir = frames_dir / label
            if not label_dir.exists():
                print(f"⚠️  Warning: {label} frames directory not found")
                continue
                
            print(f"\n👤 Processing {label} frames for face extraction...")
            
            video_dirs = [d for d in label_dir.iterdir() if d.is_dir()]
            
            if not video_dirs:
                print(f"⚠️  No video directories found in {label_dir}")
                continue
            
            for video_dir in tqdm(video_dirs, desc=f"  {label}"):
                video_name = video_dir.name
                output_video_dir = output_dir / label / video_name
                output_video_dir.mkdir(parents=True, exist_ok=True)
                
                frame_files = sorted(video_dir.glob("*.jpg"))
                
                for frame_file in frame_files:
                    face = self.extract_face(frame_file)
                    
                    if face is not None:
                        # Convert tensor to image and save
                        face_np = face.permute(1, 2, 0).numpy()
                        face_np = (face_np * 255).astype(np.uint8)
                        
                        output_path = output_video_dir / frame_file.name
                        cv2.imwrite(
                            str(output_path),
                            cv2.cvtColor(face_np, cv2.COLOR_RGB2BGR)
                        )
            
            print(f"  ✓ Completed {label} face extraction")
    
    def run(self):
        """Run the complete face extraction pipeline"""
        frames_dir = Path(self.config['dataset']['processed_data_path']) / 'frames'
        faces_dir = Path(self.config['dataset']['processed_data_path']) / 'faces'
        
        if not frames_dir.exists():
            print(f"❌ Error: Frames directory not found: {frames_dir}")
            print("   Please run frame extraction first")
            return
        
        self.process_frames(frames_dir, faces_dir)
        
        print("\n✅ Face extraction completed!")

if __name__ == "__main__":
    extractor = FaceExtractor()
    extractor.run()