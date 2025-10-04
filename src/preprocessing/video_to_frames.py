import cv2
import os
from pathlib import Path
import yaml
from tqdm import tqdm
import numpy as np

class VideoFrameExtractor:
    """Extract frames from video files"""
    
    def __init__(self, config_path='config.yaml'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.frames_per_video = self.config['dataset']['frames_per_video']
        self.fps = self.config['dataset']['fps']
        
    def extract_frames(self, video_path, max_frames=None):
        """Extract frames from a video file"""
        if max_frames is None:
            max_frames = self.frames_per_video
            
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Calculate frame interval
        if video_fps > 0:
            frame_interval = max(1, int(video_fps / self.fps))
        else:
            frame_interval = 1
        
        frames = []
        frame_count = 0
        saved_count = 0
        
        while cap.isOpened() and saved_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_count % frame_interval == 0:
                frames.append(frame)
                saved_count += 1
                
            frame_count += 1
        
        cap.release()
        return frames
    
    def process_dataset(self, video_dir, output_dir, label):
        """Process all videos in a directory"""
        video_dir = Path(video_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Support multiple video formats
        video_files = []
        for ext in ['*.mp4', '*.avi', '*.mov', '*.mkv', '*.MP4', '*.AVI', '*.MOV']:
            video_files.extend(video_dir.glob(ext))
        
        if not video_files:
            print(f"⚠️  Warning: No videos found in {video_dir}")
            return
        
        print(f"\n📹 Processing {len(video_files)} {label} videos...")
        
        successful = 0
        failed = 0
        
        for video_path in tqdm(video_files, desc=f"  {label}"):
            video_name = video_path.stem
            video_output_dir = output_dir / label / video_name
            video_output_dir.mkdir(parents=True, exist_ok=True)
            
            try:
                frames = self.extract_frames(video_path)
                
                # Save frames
                for idx, frame in enumerate(frames):
                    frame_path = video_output_dir / f"frame_{idx:04d}.jpg"
                    cv2.imwrite(str(frame_path), frame)
                
                successful += 1
                    
            except Exception as e:
                failed += 1
                print(f"\n  ⚠️  Error processing {video_path.name}: {str(e)}")
                continue
        
        print(f"  ✓ Success: {successful}, Failed: {failed}")
    
    def run(self):
        """Run the complete frame extraction pipeline"""
        real_dir = Path(self.config['dataset']['real_videos'])
        fake_dir = Path(self.config['dataset']['fake_videos'])
        output_dir = Path(self.config['dataset']['processed_data_path']) / 'frames'
        
        # Check if directories exist
        if not real_dir.exists():
            print(f"❌ Error: Real videos directory not found: {real_dir}")
            return
        
        if not fake_dir.exists():
            print(f"❌ Error: Fake videos directory not found: {fake_dir}")
            return
        
        # Process real videos
        self.process_dataset(real_dir, output_dir, 'real')
        
        # Process fake videos
        self.process_dataset(fake_dir, output_dir, 'fake')
        
        print("\n✅ Frame extraction completed!")

if __name__ == "__main__":
    extractor = VideoFrameExtractor()
    extractor.run()