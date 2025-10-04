import torch
import torch.nn as nn
from pathlib import Path
import yaml
from tqdm import tqdm
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns

class ModelEvaluator:
    """Model evaluation class"""
    
    def __init__(self, model, test_loader, config_path='config.yaml'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.model = model
        self.test_loader = test_loader
        
        self.device = torch.device(
            self.config['hardware']['device'] 
            if torch.cuda.is_available() else 'cpu'
        )
        self.model.to(self.device)
        self.model.eval()
        
    def evaluate(self):
        """Evaluate the model on test set"""
        all_labels = []
        all_predictions = []
        all_probabilities = []
        
        print("\n🔍 Evaluating model on test set...")
        
        with torch.no_grad():
            for inputs, labels in tqdm(self.test_loader, desc='Testing', ncols=100):
                inputs = inputs.to(self.device)
                
                outputs = self.model(inputs)
                probabilities = torch.softmax(outputs, dim=1)
                _, predicted = outputs.max(1)
                
                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())
                all_probabilities.extend(probabilities[:, 1].cpu().numpy())
        
        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_predictions)
        precision = precision_score(all_labels, all_predictions, zero_division=0)
        recall = recall_score(all_labels, all_predictions, zero_division=0)
        f1 = f1_score(all_labels, all_predictions, zero_division=0)
        
        try:
            auc = roc_auc_score(all_labels, all_probabilities)
        except:
            auc = 0.0
        
        # Confusion matrix
        cm = confusion_matrix(all_labels, all_predictions)
        
        # Print results
        print("\n" + "="*70)
        print("📊 EVALUATION RESULTS")
        print("="*70)
        print(f"  Accuracy:  {accuracy*100:.2f}%")
        print(f"  Precision: {precision*100:.2f}%")
        print(f"  Recall:    {recall*100:.2f}%")
        print(f"  F1 Score:  {f1*100:.2f}%")
        print(f"  AUC-ROC:   {auc*100:.2f}%")
        print("\n  Confusion Matrix:")
        print(f"    True Negatives (Real → Real):   {cm[0][0]:>6}")
        print(f"    False Positives (Real → Fake):  {cm[0][1]:>6}")
        print(f"    False Negatives (Fake → Real):  {cm[1][0]:>6}")
        print(f"    True Positives (Fake → Fake):   {cm[1][1]:>6}")
        
        # Classification report
        print("\n" + "─"*70)
        print("  Detailed Classification Report:")
        print("─"*70)
        report = classification_report(
            all_labels, all_predictions,
            target_names=['Real', 'Fake'],
            digits=4
        )
        for line in report.split('\n'):
            if line.strip():
                print(f"  {line}")
        print("="*70)
        
        # Plot confusion matrix
        self.plot_confusion_matrix(cm)
        
        # Plot ROC curve
        if auc > 0:
            self.plot_roc_curve(all_labels, all_probabilities, auc)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc,
            'confusion_matrix': cm,
            'labels': all_labels,
            'predictions': all_predictions,
            'probabilities': all_probabilities
        }
    
    def plot_confusion_matrix(self, cm):
        """Plot confusion matrix"""
        plt.figure(figsize=(10, 8))
        
        # Create heatmap
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Real', 'Fake'],
            yticklabels=['Real', 'Fake'],
            cbar_kws={'label': 'Count'},
            annot_kws={'size': 16, 'weight': 'bold'}
        )
        
        plt.title('Confusion Matrix - Deepfake Detection', fontsize=16, weight='bold', pad=20)
        plt.ylabel('True Label', fontsize=14, weight='bold')
        plt.xlabel('Predicted Label', fontsize=14, weight='bold')
        plt.tight_layout()
        
        # Save
        output_dir = Path(self.config['inference']['output_path'])
        output_dir.mkdir(exist_ok=True, parents=True)
        save_path = output_dir / 'confusion_matrix.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\n  ✓ Confusion matrix saved: {save_path}")
    
    def plot_roc_curve(self, labels, probabilities, auc):
        """Plot ROC curve"""
        fpr, tpr, thresholds = roc_curve(labels, probabilities)
        
        plt.figure(figsize=(10, 8))
        
        # Plot ROC curve
        plt.plot(
            fpr, tpr, 
            label=f'ROC Curve (AUC = {auc:.4f})', 
            linewidth=3, 
            color='#2E86AB'
        )
        
        # Plot diagonal (random classifier)
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier', linewidth=2)
        
        # Fill area under curve
        plt.fill_between(fpr, tpr, alpha=0.2, color='#2E86AB')
        
        plt.xlabel('False Positive Rate', fontsize=14, weight='bold')
        plt.ylabel('True Positive Rate', fontsize=14, weight='bold')
        plt.title('ROC Curve - Deepfake Detection', fontsize=16, weight='bold', pad=20)
        plt.legend(fontsize=12, loc='lower right')
        plt.grid(True, alpha=0.3, linestyle='--')
        plt.tight_layout()
        
        # Save
        output_dir = Path(self.config['inference']['output_path'])
        save_path = output_dir / 'roc_curve.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ ROC curve saved: {save_path}")

def main():
    """Main evaluation function"""
    config_path = 'config.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print("\n" + "="*70)
    print("📊 MODEL EVALUATION")
    print("="*70)
    
    # Load data
    from src.training.dataset import create_data_loaders
    print("\n📂 Loading test data...")
    _, _, test_loader = create_data_loaders(config_path)
    
    # Load best model
    checkpoint_path = Path(config['training']['checkpoint_dir']) / 'best_model.pth'
    
    if not checkpoint_path.exists():
        print(f"\n❌ Error: Model checkpoint not found at {checkpoint_path}")
        print("   Please train the model first using: python main.py --mode train")
        return
    
    print(f"\n🧠 Loading trained model from: {checkpoint_path}")
    
    architecture = config['model']['architecture']
    
    if 'efficientnet' in architecture:
        from src.models.efficientnet_model import load_pretrained_efficientnet
        model = load_pretrained_efficientnet(
            model_name=architecture,
            num_classes=config['model']['num_classes']
        )
    else:
        from src.models.xception_model import load_pretrained_xception
        model = load_pretrained_xception(num_classes=config['model']['num_classes'])
    
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"  ✓ Model loaded (Epoch {checkpoint['epoch'] + 1})")
    print(f"  ✓ Best validation loss: {checkpoint['val_loss']:.4f}")
    print(f"  ✓ Best validation accuracy: {checkpoint['val_acc']:.2f}%")
    
    # Evaluate
    evaluator = ModelEvaluator(model, test_loader, config_path)
    results = evaluator.evaluate()
    
    print("\n✅ Evaluation completed!")
    print(f"   Results saved to: {config['inference']['output_path']}\n")

if __name__ == "__main__":
    main()