"""
Hierarchical Multi-task Learning - Training Script
"""
import os
import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from sklearn.metrics import f1_score

from settings import (
    TrainConfig, ModelConfig, LogConfig,
    ORIENTATION_MODEL, COMPONENT_MODELS,
    FLOOD_GRADES, HIER_MULT_DIR
)
from dataset import FloodDataset, create_dataloader
from model import HierarchicalFloodModel
from helpers import (
    set_random_seed, create_run_directory, save_json,
    compute_class_weights, plot_training_curves, plot_confusion_matrix,
    generate_classification_report, AverageMeter, EarlyStopping,
    format_time, print_model_summary
)


class Trainer:
    """Trainer"""
    
    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
        
        # Set random seed
        set_random_seed(args.seed)
        
        # Create run directory
        self.run_dir = create_run_directory(args.runs_dir)
        print(f"\n📁 Run directory: {self.run_dir}")
        
        # Saving config
        config_dict = dict(vars(args))  # Convert to regular dict
        save_json(config_dict, os.path.join(self.run_dir, 'config.json'))
        
        # Build dataset
        print("\n" + "=" * 70)
        print("Loading dataset")
        print("=" * 70)
        # Prepare oversampling config
        oversample_rates = {
            0: args.oversample_level0,
            1: args.oversample_level1
        } if args.oversample else None
        
        self.train_dataset = FloodDataset(
            csv_path=args.train_csv,
            image_size=args.image_size,
            is_train=True,
            augment=args.augment,
            oversample=args.oversample,
            oversample_rates=oversample_rates
        )
        self.val_dataset = FloodDataset(
            csv_path=args.val_csv,
            image_size=args.image_size,
            is_train=False,
            augment=False
        )
        
        # Data loader
        self.train_loader = create_dataloader(
            self.train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory
        )
        self.val_loader = create_dataloader(
            self.val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory
        )
        
        print(f"Train set: {len(self.train_dataset)} samples")
        print(f"Validation set: {len(self.val_dataset)} samples")
        print(f"Train batches: {len(self.train_loader)}")
        print(f"Validation batches: {len(self.val_loader)}")
        
        # Compute class weights
        if args.use_class_weights:
            train_labels = [int(item['flood_grade'].item()) 
                          for item in self.train_dataset]
            self.class_weights = compute_class_weights(train_labels, 5).to(self.device)  # type: ignore
            print(f"\nClass weights: {self.class_weights.cpu().numpy()}")
        else:
            self.class_weights = None
        
        # Build model
        print("\n" + "=" * 70)
        print("Building model")
        print("=" * 70)
        
        # Convert model paths to strings
        component_models_dict = {k: str(v) for k, v in COMPONENT_MODELS.items()}
        
        self.model = HierarchicalFloodModel(
            orientation_model_path=str(ORIENTATION_MODEL),
            component_model_paths=component_models_dict,
            num_flood_classes=5,
            freeze_backbone=args.freeze_backbone,
            use_component_branch=args.use_component_branch
        ).to(self.device)
        
        print_model_summary(self.model)
        
        # Optimizer
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=args.learning_rate,
            weight_decay=args.weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=args.epochs,
            eta_min=args.learning_rate * 0.01
        )
        
        # Mixed precision training
        self.use_amp = args.mixed_precision and self.device.type == 'cuda'
        if self.use_amp:
            self.scaler = GradScaler()
            print("✓ Mixed precision training enabled")
        
        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=args.patience,
            min_delta=args.min_delta,
            mode='min'
        )
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_flood_loss': [],
            'val_flood_loss': [],
            'val_accuracy': [],
            'learning_rate': [],
        }
        
        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0
        self.start_epoch = 1
        
        # Resume training (if specified)
        if args.resume:
            self.load_checkpoint(args.resume)
    
    def train_epoch(self, epoch: int):
        """Train one epoch"""
        self.model.train()
        
        losses = AverageMeter('Loss')
        flood_losses = AverageMeter('Flood Loss')
        
        all_preds = []
        all_targets = []
        
        start_time = time.time()
        
        for batch_idx, batch in enumerate(self.train_loader):
            images = batch['image'].to(self.device)
            targets = {
                'flood_grade': batch['flood_grade'].to(self.device),
                'confidence': batch['confidence'].to(self.device)  # Add confidence
            }
            
            # Forward pass
            if self.use_amp:
                with autocast():
                    outputs = self.model(images)
                    loss, loss_dict = self.model.compute_loss(
                        outputs, targets,
                        class_weights=self.class_weights,
                        label_smoothing=self.args.label_smoothing,
                        use_confidence=True,  # Enable confidence-based dynamic label smoothing
                        use_focal_loss=self.args.use_focal_loss,  # Use Focal Loss
                        focal_gamma=self.args.focal_gamma
                    )
            else:
                outputs = self.model(images)
                loss, loss_dict = self.model.compute_loss(
                    outputs, targets,
                    class_weights=self.class_weights,
                    label_smoothing=self.args.label_smoothing,
                    use_confidence=True,  # Enable confidence-based dynamic label smoothing
                    use_focal_loss=self.args.use_focal_loss,  # Use Focal Loss
                    focal_gamma=self.args.focal_gamma
                )
            
            # Backward pass
            self.optimizer.zero_grad()
            if self.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
            
            # Record
            losses.update(loss.item(), images.size(0))
            flood_losses.update(loss_dict['flood_loss'], images.size(0))
            
            # Collect predictions (for F1 calculation)
            with torch.no_grad():
                preds = torch.argmax(outputs['flood_grade'], dim=1)
                all_preds.extend(preds.cpu().numpy().tolist())
                all_targets.extend(targets['flood_grade'].cpu().numpy().tolist())
            
            # Print
            if (batch_idx + 1) % self.args.print_freq == 0:
                lr = self.optimizer.param_groups[0]['lr']
                elapsed = time.time() - start_time
                print(f"Epoch [{epoch}/{self.args.epochs}] "
                      f"Batch [{batch_idx+1}/{len(self.train_loader)}] "
                      f"Loss: {losses.avg:.4f} | "
                      f"LR: {lr:.2e} | "
                      f"Time: {format_time(elapsed)}")
        
        # Compute train accuracy and F1
        train_acc = sum(p == t for p, t in zip(all_preds, all_targets)) / len(all_targets)
        train_f1 = f1_score(all_targets, all_preds, average='macro', zero_division='warn')
        
        return losses.avg, flood_losses.avg, train_acc, train_f1
    
    @torch.no_grad()
    def validate(self, epoch: int):
        """Validate"""
        self.model.eval()
        
        losses = AverageMeter('Val Loss')
        flood_losses = AverageMeter('Val Flood Loss')
        
        all_preds = []
        all_targets = []
        
        for batch in self.val_loader:
            images = batch['image'].to(self.device)
            targets = {
                'flood_grade': batch['flood_grade'].to(self.device),
                'confidence': batch['confidence'].to(self.device)  # Add confidence
            }
            
            # Forward pass
            outputs = self.model(images)
            loss, loss_dict = self.model.compute_loss(
                outputs, targets,
                class_weights=self.class_weights,
                label_smoothing=0.0  # Validation does not use label smoothing
            )
            
            # Record
            losses.update(loss.item(), images.size(0))
            flood_losses.update(loss_dict['flood_loss'], images.size(0))
            
            # Prediction
            preds = torch.argmax(outputs['flood_grade'], dim=1)
            all_preds.extend(preds.cpu().numpy().tolist())
            all_targets.extend(targets['flood_grade'].cpu().numpy().tolist())
        
        # Compute accuracy and F1
        correct = sum(p == t for p, t in zip(all_preds, all_targets))
        accuracy = correct / len(all_targets)
        val_f1 = f1_score(all_targets, all_preds, average='macro', zero_division='warn')
        
        return losses.avg, flood_losses.avg, accuracy, val_f1, all_preds, all_targets
    
    def save_checkpoint(self, filename: str, is_best: bool = False):
        """Save checkpoint"""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'best_val_acc': self.best_val_acc,
            'history': self.history,
            'early_stopping': self.early_stopping.state_dict(),
        }
        
        if self.use_amp:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        filepath = os.path.join(self.run_dir, filename)
        torch.save(checkpoint, filepath)
        
        if is_best:
            best_path = os.path.join(self.run_dir, 'best_model.pth')
            torch.save(checkpoint, best_path)
            print(f"✅ Best model saved: {best_path}")
    
    def load_checkpoint(self, filepath: str):
        """Load checkpoint"""
        print(f"Loading checkpoint: {filepath}")
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.best_val_loss = checkpoint['best_val_loss']
        self.best_val_acc = checkpoint['best_val_acc']
        self.history = checkpoint['history']
        self.start_epoch = checkpoint['epoch'] + 1
        self.early_stopping.load_state_dict(checkpoint['early_stopping'])
        
        if self.use_amp and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        print(f"Resuming training from epoch {checkpoint['epoch']}")
    
    def train(self):
        """Main training loop"""
        print("\n" + "=" * 70)
        print("Start training")
        print("=" * 70)
        
        total_start_time = time.time()
        
        for epoch in range(self.start_epoch, self.args.epochs + 1):
            self.current_epoch = epoch
            epoch_start_time = time.time()
            
            # Training
            train_loss, train_flood_loss, train_acc, train_f1 = self.train_epoch(epoch)
            
            # Validation
            val_loss, val_flood_loss, val_acc, val_f1, val_preds, val_targets = self.validate(epoch)
            
            # Learning rate scheduling
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Record history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_flood_loss'].append(train_flood_loss)
            self.history['val_flood_loss'].append(val_flood_loss)
            self.history['val_accuracy'].append(val_acc)
            self.history['learning_rate'].append(current_lr)
            
            # Print table
            epoch_time = time.time() - epoch_start_time
            print(f"\n{'='*70}")
            print(f"EPOCH [{epoch}/{self.args.epochs}]")
            print(f"{'='*70}")
            print(f"{'Metric':<20} {'Train Set':<15} {'Validation Set':<20}")
            print(f"{'-'*70}")
            print(f"{'Total Loss':<20} {train_loss:<15.4f} {val_loss:<20.4f}")
            print(f"{'Macro F1':<20} {train_f1:<15.4f} {val_f1:<20.4f} <-- focus")
            print(f"{'Accuracy':<20} {train_acc:<15.4f} {val_acc:<20.4f}")
            print(f"{'='*70}")
            print(f"Learning rate: {current_lr:.2e} | Time: {format_time(epoch_time)}")
            print(f"{'='*70}\n")
            
            # Save best model
            is_best = False
            old_best_val_loss = self.best_val_loss
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                is_best = True
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                is_best = True
            
            if is_best:
                self.save_checkpoint('best_model.pth', is_best=True)
                if old_best_val_loss != float('inf'):
                    print(f">> New Best Model Saved: Val Loss improved from {old_best_val_loss:.4f} to {val_loss:.4f}.")
                else:
                    print(f">> New Best Model Saved: Val Loss = {val_loss:.4f}.")
            
            # Save periodically
            if epoch % self.args.save_interval == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch}.pth')
            
            # Save latest
            self.save_checkpoint('last_model.pth')
            
            # Save training curves and history
            plot_training_curves(self.history, 
                               os.path.join(self.run_dir, 'training_curves.png'))
            save_json(self.history, 
                     os.path.join(self.run_dir, 'training_history.json'))
            
            # Early stopping check
            should_stop = self.early_stopping(val_loss)
            patience_counter = self.early_stopping.counter
            print(f">> Patience: {patience_counter}/{self.args.patience}")
            
            if should_stop:
                print(f"\n⚠️  Early stopping triggered! Stopping training at epoch {epoch}")
                break
        
        # Training complete
        total_time = time.time() - total_start_time
        print("\n" + "=" * 70)
        print("Training complete!")
        print("=" * 70)
        print(f"Total time: {format_time(total_time)}")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        print(f"Best validation accuracy: {self.best_val_acc:.4f} ({self.best_val_acc*100:.2f}%)")
        print(f"Model saved at: {self.run_dir}")
        
        # Final evaluation
        self.final_evaluation(val_preds, val_targets)
    
    def final_evaluation(self, preds, targets):
        """Final evaluation"""
        print("\n" + "=" * 70)
        print("Final evaluation")
        print("=" * 70)
        
        # Confusion matrix
        class_names = [f"Level {i}\n{FLOOD_GRADES[i].capitalize()}" for i in range(5)]
        plot_confusion_matrix(targets, preds, class_names,
                            os.path.join(self.run_dir, 'confusion_matrix.png'))
        
        # Classification report
        class_names_simple = [f"{i}_{FLOOD_GRADES[i].capitalize()}" for i in range(5)]
        generate_classification_report(targets, preds, class_names_simple,
                                      os.path.join(self.run_dir, 'classification_report.txt'))


def parse_args():
    parser = argparse.ArgumentParser(description='Train hierarchical multi-task flood level assessment model')
    
    # Data
    parser.add_argument('--train_csv', type=str, 
                       default=str(HIER_MULT_DIR / 'data' / 'train_with_confidence.csv'),
                       help='Train CSV file path')
    parser.add_argument('--val_csv', type=str, 
                       default=str(HIER_MULT_DIR / 'data' / 'val_with_confidence.csv'),
                       help='Validation CSV file path')
    parser.add_argument('--image_size', type=int, default=TrainConfig.image_size,
                       help='Image size')
    parser.add_argument('--augment', type=bool, default=True,
                       help='Whether to use data augmentation')
    
    # Training
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=TrainConfig.learning_rate,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=TrainConfig.weight_decay,
                       help='Weight decay')
    parser.add_argument('--label_smoothing', type=float, default=TrainConfig.label_smoothing,
                       help='Label smoothing')
    parser.add_argument('--use_class_weights', type=bool, default=TrainConfig.use_class_weights,
                       help='Whether to use class weights')
    parser.add_argument('--use_focal_loss', type=bool, default=True,
                       help='Whether to use Focal Loss (recommended for class imbalance)')
    parser.add_argument('--focal_gamma', type=float, default=1.5,
                       help='Gamma parameter for Focal Loss (reduces over-focus on hard samples)')
    parser.add_argument('--oversample', type=bool, default=True,
                       help='Whether to apply oversampling to minority classes')
    parser.add_argument('--oversample_level0', type=int, default=3,
                       help='Oversampling factor for Level 0')
    parser.add_argument('--oversample_level1', type=int, default=1,
                       help='Oversampling factor for Level 1')
    
    # Model
    parser.add_argument('--freeze_backbone', type=bool, default=ModelConfig.freeze_orientation_model,
                       help='Whether to freeze the backbone network')
    parser.add_argument('--use_component_branch', type=bool, default=ModelConfig.use_component_branch,
                       help='Whether to use the component branch')
    
    # Optimization
    parser.add_argument('--mixed_precision', type=bool, default=TrainConfig.mixed_precision,
                       help='Whether to use mixed precision training')
    parser.add_argument('--patience', type=int, default=TrainConfig.patience,
                       help='Early stopping patience')
    parser.add_argument('--min_delta', type=float, default=TrainConfig.min_improvement,
                       help='Minimum improvement for early stopping')
    
    # System
    parser.add_argument('--device', type=str, default=TrainConfig.device,
                       choices=['cuda', 'cpu'], help='Device')
    parser.add_argument('--num_workers', type=int, default=TrainConfig.num_workers,
                       help='Number of data loader workers')
    parser.add_argument('--pin_memory', type=bool, default=TrainConfig.pin_memory,
                       help='Whether to pin memory')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    # Saving
    parser.add_argument('--runs_dir', type=str, default=str(HIER_MULT_DIR / 'runs'),
                       help='Runs directory')
    parser.add_argument('--save_interval', type=int, default=TrainConfig.save_every_n_epochs,
                       help='Save interval (epochs)')
    parser.add_argument('--print_freq', type=int, default=LogConfig.print_freq,
                       help='Print frequency (batch)')
    
    # Resume training
    parser.add_argument('--resume', type=str, default=None,
                       help='Checkpoint path to resume training')
    
    return parser.parse_args()


def main():
    args = parse_args()
    trainer = Trainer(args)
    trainer.train()


if __name__ == '__main__':
    main()

