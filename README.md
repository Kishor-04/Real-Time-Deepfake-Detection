# 🎯 Real-Time Deepfake Detection (Model A)

**Author:** Kishor Yogesh Patil (Kishor-04)  
**Project Type:** Video Deepfake Detection using Pre-trained Models  
**Hardware:** NVIDIA RTX 4070 GPU  
**Dataset:** 500 real + 474 fake videos (974 total)

---

## 📋 Overview

This project implements a **fully pre-trained model approach (Model A)** for video deepfake detection. It uses EfficientNet-B0 pre-trained on ImageNet, then fine-tuned on our custom dataset of real and fake videos.

### ✨ Features

- ✅ **Pre-trained Model:** EfficientNet-B0 with ImageNet weights
- ✅ **Face Detection:** Automatic face extraction using MTCNN
- ✅ **Data Augmentation:** Advanced augmentation techniques
- ✅ **GPU Optimized:** Fully optimized for RTX 4070
- ✅ **TensorBoard Support:** Real-time training monitoring
- ✅ **Video Inference:** Batch and single video prediction

---

## 🚀 Quick Start

### 1. Clone Repository
```bash
git clone https://github.com/Kishor-04/Real-Time-Deepfake-Detection.git
cd Real-Time-Deepfake-Detection
```

### 2. Setup Environment
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Verify GPU
```bash
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}')"
```

### 4. Prepare Dataset
```bash
# Place your videos in these folders:
# - data/raw/real/  (500 real videos)
# - data/raw/fake/  (474 fake videos)
```

### 5. Run Complete Pipeline
```bash
python main.py --mode all
```

---

## 📊 Usage

### Run Individual Steps

```bash
# Step 1: Extract frames and faces (one-time, slow)
python main.py --mode preprocess

# Step 2: Prepare dataset (randomize + split) (one-time, fast)
python main.py --mode prepare_dataset

# Step 3: Train model (can repeat with different configs)
python main.py --mode train

# Step 4: Evaluate model
python main.py --mode evaluate

# Step 5: Inference on new videos
python main.py --mode inference --video path/to/video.mp4
python main.py --mode inference --batch path/to/videos/
```

### Monitor Training with TensorBoard
```bash
tensorboard --logdir runs/deepfake_detection
```

---

## 📈 Expected Results

- **Training Time:** 4-6 hours (RTX 4070)
- **Accuracy:** 90-95%
- **Inference Speed:** 5-8 seconds per video
- **Model Size:** ~20 MB

---

## 📁 Project Structure

```
Real-Time-Deepfake-Detection/
├── README.md
├── requirements.txt
├── config.yaml
├── main.py
├── .gitignore
│
├── data/
│   ├── raw/
│   │   ├── real/              # Place your 500 real videos here
│   │   └── fake/              # Place your 474 fake videos here
│   └── processed/
│       ├── frames/
│       │   ├── real/
│       │   └── fake/
│       ├── faces/
│       │   ├── real/
│       │   │   ├── 0/             
│       │   │   ├── 1/
│       │   │   └── ...
│       │   ├── fake/
│       │   │   ├── 0/
│       │   │   └── ...
│       │   ├── real_folder_mapping.txt 
│       │   └── fake_folder_mapping.txt  
│       │
│       └── splits/                
│           └── data_splits.pkl    
│
├── models/
│   ├── pretrained/            # Pre-trained weights (optional)
│   └── checkpoints/           # Your trained models will be saved here
│
├── src/
│   ├── __init__.py
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   ├── video_to_frames.py
│   │   ├── face_extraction.py
│   │   └── data_augmentation.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── xception_model.py
│   │   └── efficientnet_model.py
│   ├── training/
│   │   ├── __init__.py
│   │   ├── dataset.py
│   │   ├── train.py
│   │   └── fine_tune.py
│   ├── evaluation/
│   │   ├── __init__.py
│   │   └── evaluate.py
│   └── inference/
│       ├── __init__.py
│       └── predict_video.py
│
├── scripts/
│   ├── download_weights.py
│   └── setup_environment.sh
│
├── notebooks/
│   └── exploration.ipynb
│
├── results/
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   └── predictions.csv
│
└── runs/                      # TensorBoard logs
    └── deepfake_detection/
```

---

## 🛠️ Configuration

Edit `config.yaml` to customize:
- Model architecture (EfficientNet-B0, Xception, etc.)
- Training hyperparameters
- Data augmentation settings
- Hardware configuration

---

## 📝 License

MIT License - Feel free to use for academic and commercial purposes.

---

## 👨‍💻 Author

**Kishor Yogesh Patil**  
- GitHub: [@Kishor-04](https://github.com/Kishor-04)
- Portfolio: [kishorpatil.me](https://kishorpatil.me/)
- Location: Nashik, India

Final-year IT student and Full Stack Developer (MERN stack)

---

## 🙏 Acknowledgments

- EfficientNet: [efficientnet-pytorch](https://github.com/lukemelas/EfficientNet-PyTorch)
- MTCNN: [facenet-pytorch](https://github.com/timesler/facenet-pytorch)
- Dataset inspiration: FaceForensics++, Celeb-DF

---