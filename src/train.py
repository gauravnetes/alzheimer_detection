import torch
import torch.optim as optim
from tqdm import tqdm # progress bar 
import torch.nn as nn
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim.lr_scheduler import CosineAnnealingLR
import time
from . import config
from . import model as model_def
from . import dataset

def train_model(): 
    # seeding block
    torch.manual_seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)
    if torch.cuda.is_available(): 
        torch.cuda.manual_seed_all(config.RANDOM_SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    model_save_path = f"{config.MODEL_SAVE_PATH}{config.RUN_NAME}_{timestamp}.pth"
    print(f"Model will be saved to: {model_save_path}")

    # Load data
    train_loader, test_loader, _, _ = dataset.create_dataloaders()
    
    # create model 
    model = model_def.create_model() 
    
    # Loss function and Optimizer 
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.Adam([
        {'params': model.layer3.parameters(), 'lr': 1e-5}, # TINY LR for deep layers
        {'params': model.layer4.parameters(), 'lr': 5e-5}, # Low LR for later layers
        {'params': model.fc.parameters(), 'lr': 1e-3}, 
    ],         
        weight_decay=1e-4
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=config.NUM_EPOCHS, eta_min=1e-6)
    
    
    print("Training Starts...")
    
    best_val_acc = 0.0
    
    # training loop 
    for epoch in range(config.NUM_EPOCHS): 
        model.train()
        running_loss = 0.0
        running_corrects = 0
        
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1} / {config.NUM_EPOCHS} [Training]"): 
            
            inputs, labels = inputs.to(config.DEVICE), labels.to(config.DEVICE)
            
            # params gradient -> 0
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            
            # backward pass and optimization 
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.double() / len(train_loader.sampler)

        print(f"Train Loss: {epoch_loss: .4f} Acc: {epoch_acc: .4f}")
        
        # validation 
        model.eval()
        val_loss = 0.0
        val_corrects = 0
        
        with torch.no_grad(): 
            for inputs, labels in tqdm(test_loader, desc=f"Epoch {epoch+1} / {config.NUM_EPOCHS} [Validation]"): 
                
                inputs, labels = inputs.to(config.DEVICE), labels.to(config.DEVICE)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * inputs.size(0)
                val_corrects += torch.sum(preds == labels.data)
        
        val_epoch_loss = val_loss / len(test_loader.dataset)
        val_epoch_acc = val_corrects.double() / len(test_loader.dataset)
        
        print(f"Validation Loss: {val_epoch_loss: .4f} Acc: {val_epoch_acc: .4f}")
        scheduler.step()
        print('-' * 20)
        
        if val_epoch_acc > best_val_acc: 
            best_val_acc = val_epoch_acc
            torch.save(model.state_dict(), model_save_path)
            print(f"New best model saved to {model_save_path} with accuracy: {best_val_acc: .4f}")
            

if __name__ == '__main__': 
    train_model()  