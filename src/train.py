import torch
import torch.optim as optim
from tqdm import tqdm
import torch.nn as nn
import numpy as np
import time
import os
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts, CosineAnnealingLR
from collections import defaultdict
import matplotlib.pyplot as plt
from . import config
from . import model as model_def
from . import dataset

# --- HELPER CLASSES AND FUNCTIONS (Now driven by config) ---

class EarlyStopping:
    """Early stopping to prevent overfitting, configured from config.py."""
    def __init__(self):
        self.patience = config.TRAINING_CONFIG['early_stopping_patience']
        self.min_delta = config.TRAINING_CONFIG['early_stopping_min_delta']
        self.counter = 0
        self.best_loss = None
        
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
        return self.counter >= self.patience

def mixup_data(x, y, alpha=1.0):
    """Mixup augmentation logic."""
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Mixup loss calculation."""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def plot_training_history(history, model_name, timestamp):
    """Plot and save training and validation metrics."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    ax1.plot(history['train_loss'], label='Train Loss')
    ax1.plot(history['val_loss'], label='Val Loss')
    ax1.set_title('Model Loss')
    ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss'); ax1.legend()
    
    ax2.plot(history['train_acc'], label='Train Acc')
    ax2.plot(history['val_acc'], label='Val Acc')
    ax2.set_title('Model Accuracy')
    ax2.set_xlabel('Epoch'); ax2.set_ylabel('Accuracy'); ax2.legend()
    
    plt.tight_layout()
    save_path = f"reports/{config.RUN_NAME}_{model_name}_{timestamp}_history.png"
    plt.savefig(save_path)
    print(f"Training history plot saved to {save_path}")

# --- CORE TRAINING AND VALIDATION EPOCHS ---

def train_epoch(model, data_loader, criterion, optimizer, device, cfg):
    model.train()
    running_loss, running_corrects, total_samples = 0.0, 0.0, 0
    
    for inputs, labels in tqdm(data_loader, desc="Training"):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        
        if cfg['use_mixup'] and np.random.random() < cfg['mixup_prob']:
            inputs, targets_a, targets_b, lam = mixup_data(inputs, labels, alpha=cfg['mixup_alpha'])
            outputs = model(inputs)
            loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
            _, preds = torch.max(outputs, 1)
            running_corrects += lam * (preds == targets_a.data).sum() + (1 - lam) * (preds == targets_b.data).sum()
        else:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels.data)
            
        loss.backward()
        if cfg.get('gradient_clip_norm'):
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg['gradient_clip_norm'])
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        total_samples += inputs.size(0)
        
    return running_loss / total_samples, running_corrects.double() / total_samples

def validate_epoch(model, data_loader, criterion, device):
    model.eval()
    running_loss, running_corrects, total_samples = 0.0, 0.0, 0
    
    with torch.no_grad():
        for inputs, labels in tqdm(data_loader, desc="Validation"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)
            
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            total_samples += inputs.size(0)
            
    return running_loss / total_samples, running_corrects.double() / total_samples

# --- MAIN TRAINING WORKFLOW ---

def train_model():
    cfg = config.TRAINING_CONFIG
    model_name = cfg['model_name']
    
    # --- Seeding for reproducibility ---
    torch.manual_seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)
    
    # --- Setup model save path ---
    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    model_save_path = f"{config.MODEL_SAVE_PATH}{config.RUN_NAME}_{model_name}_{timestamp}.pth"
    print(f"Model will be saved to: {model_save_path}")

    # --- Data, Model, Loss ---
    train_loader, test_loader, _, _ = dataset.create_dataloaders()
    model = model_def.create_model()
    
    if cfg['use_focal_loss']:
        criterion = model_def.FocalLoss(alpha=cfg['focal_alpha'], gamma=cfg['focal_gamma'])
        print("Using Focal Loss.")
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=cfg['label_smoothing'])
        print("Using Cross Entropy Loss with Label Smoothing.")

    # --- Dynamic Optimizer Setup ---
    # This requires a more complex function if you want to use the named multipliers.
    # For now, we create a simplified version. A full version would map model layers to names.
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['base_lr'], weight_decay=cfg['weight_decay'])
    print(f"Using AdamW optimizer with base LR {cfg['base_lr']}.")

    # --- Dynamic Scheduler Setup ---
    if cfg['lr_scheduler'] == 'cosine_warm_restarts':
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=cfg['min_lr'])
    elif cfg['lr_scheduler'] == 'reduce_lr_on_plateau':
        scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    else: # Default to cosine
        scheduler = CosineAnnealingLR(optimizer, T_max=config.NUM_EPOCHS, eta_min=cfg['min_lr'])
    
    early_stopping = EarlyStopping()
    history = defaultdict(list)
    best_val_metric = 0.0
    
    # --- Main Training Loop ---
    for epoch in range(config.NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{config.NUM_EPOCHS}")
        
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, config.DEVICE, cfg)
        val_loss, val_acc = validate_epoch(model, test_loader, criterion, config.DEVICE)
        
        if isinstance(scheduler, ReduceLROnPlateau):
            scheduler.step(val_acc)
        else:
            scheduler.step()

        # Store history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc.cpu())
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc.cpu())
        
        print(f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")
        
        # Save best model based on the chosen metric
        current_metric = val_acc if cfg['monitor_metric'] == 'val_accuracy' else -val_loss
        if current_metric > best_val_metric:
            best_val_metric = current_metric
            torch.save(model.state_dict(), model_save_path)
            print(f"New best model saved with {cfg['monitor_metric']}: {current_metric:.4f}")
            
        if early_stopping(val_loss):
            print(f"Early stopping at epoch {epoch+1}")
            break
            
    if cfg['save_training_plots']:
        plot_training_history(history, model_name, timestamp)
        
    print(f"\nTraining completed! Best validation metric: {best_val_metric:.4f}")
    return model_save_path, best_val_metric

if __name__ == '__main__':
    config.print_config()
    train_model()