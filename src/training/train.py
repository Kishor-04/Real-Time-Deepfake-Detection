import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import yaml
from tqdm import tqdm
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import os
import time

class DeepfakeTrainer:
    """Trainer class for deepfake detection model"""
    
    def __init__(self, model, train_loader, val_loader, config_path='config.yaml'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        # Setup device
        self.device = torch.device(
            self.config['hardware']['device'] 
            if torch.cuda.is_available() else 'cpu'
        )
        self.model.to(self.device)
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config['training']['learning_rate'],
            weight_decay=self.config['training']['weight_decay']
        )
        
        # Learning rate scheduler (FIXED - removed verbose parameter)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            patience=3,
            factor=0.5
        )
        
        # Training parameters
        self.num_epochs = self.config['training']['epochs']
        self.checkpoint_dir = Path(self.config['training']['checkpoint_dir'])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Early stopping
        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0
        self.early_stopping_patience = self.config['training']['early_stopping_patience']
        self.patience_counter = 0
        
        # TensorBoard
        self.writer = SummaryWriter('runs/deepfake_detection')
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'lr': []
        }
        
    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(
            self.train_loader, 
            desc=f'Epoch {epoch+1}/{self.num_epochs} [Train]',
            ncols=100
        )
        
        for batch_idx, (inputs, labels) in enumerate(pbar):
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{running_loss / (batch_idx + 1):.4f}',
                'acc': f'{100. * correct / total:.2f}%'
            })
        
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def validate(self, epoch):
        """Validate the model"""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            pbar = tqdm(
                self.val_loader, 
                desc=f'Epoch {epoch+1}/{self.num_epochs} [Val]  ',
                ncols=100
            )
            
            for inputs, labels in pbar:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': f'{running_loss / (len(pbar)):.4f}',
                    'acc': f'{100. * correct / total:.2f}%'
                })
        
        val_loss = running_loss / len(self.val_loader)
        val_acc = 100. * correct / total
        
        return val_loss, val_acc
    
    def save_checkpoint(self, epoch, val_loss, val_acc, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'val_acc': val_acc,
            'history': self.history
        }
        
        # Save latest checkpoint
        checkpoint_path = self.checkpoint_dir / 'latest_checkpoint.pth'
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_path = self.checkpoint_dir / 'best_model.pth'
            torch.save(checkpoint, best_path)
            print(f"  💾 Saved best model (val_loss: {val_loss:.4f}, val_acc: {val_acc:.2f}%)")
    
    def train(self):
        """Main training loop"""
        print(f"\n{'='*70}")
        print(f"🚀 TRAINING STARTED")
        print(f"{'='*70}")
        print(f"  Device: {self.device}")
        print(f"  Model: {self.config['model']['architecture']}")
        print(f"  Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"  Trainable: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")
        print(f"  Epochs: {self.num_epochs}")
        print(f"  Batch size: {self.config['training']['batch_size']}")
        print(f"  Learning rate: {self.config['training']['learning_rate']}")
        print(f"{'='*70}\n")
        
        # Freeze backbone initially if specified
        if self.config['training']['freeze_layers']:
            self.model.freeze_backbone()
            print("🔒 Backbone frozen for initial training")
            print(f"   Will unfreeze after epoch {self.config['training']['unfreeze_after_epoch']}\n")
        
        start_time = time.time()
        
        for epoch in range(self.num_epochs):
            # Unfreeze backbone after specified epochs
            if (self.config['training']['freeze_layers'] and 
                epoch == self.config['training']['unfreeze_after_epoch']):
                self.model.unfreeze_backbone()
                print(f"\n🔓 Backbone unfrozen for fine-tuning\n")
            
            # Train
            train_loss, train_acc = self.train_epoch(epoch)
            
            # Validate
            val_loss, val_acc = self.validate(epoch)
            
            # Learning rate scheduling (WITH MANUAL VERBOSE)
            old_lr = self.optimizer.param_groups[0]['lr']
            self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['lr'].append(current_lr)
            
            # Log to tensorboard
            self.writer.add_scalar('Loss/train', train_loss, epoch)
            self.writer.add_scalar('Loss/val', val_loss, epoch)
            self.writer.add_scalar('Accuracy/train', train_acc, epoch)
            self.writer.add_scalar('Accuracy/val', val_acc, epoch)
            self.writer.add_scalar('Learning_Rate', current_lr, epoch)
            
            # Print epoch results
            print(f"\n{'─'*70}")
            print(f"  Epoch {epoch+1}/{self.num_epochs} Summary:")
            print(f"  Train Loss: {train_loss:.4f}  |  Train Acc: {train_acc:.2f}%")
            print(f"  Val Loss:   {val_loss:.4f}  |  Val Acc:   {val_acc:.2f}%")
            print(f"  Learning Rate: {current_lr:.6f}")
            
            # Show LR change if it happened (MANUAL VERBOSE OUTPUT)
            if old_lr != current_lr:
                print(f"  📉 Learning rate reduced: {old_lr:.6f} → {current_lr:.6f}")
            
            # Save checkpoint
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                self.best_val_acc = val_acc
                self.patience_counter = 0
            else:
                self.patience_counter += 1
            
            self.save_checkpoint(epoch, val_loss, val_acc, is_best)
            
            # Early stopping
            if self.patience_counter >= self.early_stopping_patience:
                print(f"\n{'─'*70}")
                print(f"⚠️  Early stopping triggered after {epoch+1} epochs")
                print(f"   No improvement for {self.early_stopping_patience} epochs")
                break
            
            print(f"{'─'*70}\n")
        
        # Training completed
        elapsed_time = time.time() - start_time
        hours = int(elapsed_time // 3600)
        minutes = int((elapsed_time % 3600) // 60)
        seconds = int(elapsed_time % 60)
        
        print(f"\n{'='*70}")
        print(f"✅ TRAINING COMPLETED!")
        print(f"{'='*70}")
        print(f"  Total time: {hours}h {minutes}m {seconds}s")
        print(f"  Best val loss: {self.best_val_loss:.4f}")
        print(f"  Best val acc: {self.best_val_acc:.2f}%")
        print(f"  Model saved: {self.checkpoint_dir / 'best_model.pth'}")
        print(f"{'='*70}\n")
        
        self.writer.close()