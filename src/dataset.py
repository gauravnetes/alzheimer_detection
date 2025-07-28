import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, WeightedRandomSampler
import numpy as np
from . import config

def create_dataloaders(): 
    # create trtaining and testing dataloaders
    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(30), # Increase rotation
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        transforms.TrivialAugmentWide(),
        transforms.ToTensor(), 
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    test_transforms = transforms.Compose([
        transforms.Resize((224, 224)), 
        transforms.ToTensor(), 
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    train_dataset = datasets.ImageFolder(config.TRAIN_DIR, transform=train_transforms)
    test_dataset = datasets.ImageFolder(config.TEST_DIR, transform=test_transforms)
    
    # weighted random samplers 
    train_targets = [sample[1] for sample in train_dataset.samples]
    class_count = np.bincount(train_targets)
    class_weights = 1. / class_count # rarer classes get the higher score
    weights = class_weights[train_targets]
    sampler = WeightedRandomSampler(weights=torch.DoubleTensor(weights), num_samples=len(weights))
    
    train_loader = DataLoader(
        dataset=train_dataset, 
        batch_size=config.BATCH_SIZE, 
        sampler=sampler
    )
    
    test_loader = DataLoader(
        dataset=test_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=False    
    )
    print(f"Class counts: {class_count}")
    print("DataLoaders created Suuccessfully.")
    return train_loader, test_loader, train_dataset, test_dataset



# Add this block at the end of src/dataset.py

if __name__ == '__main__':
    # This code will only run when you execute the file directly
    print("Running dataset.py for testing...")
    # Call your function to create and test the dataloaders
    train_loader, test_loader, train_dataset, test_dataset = create_dataloaders()
    
    # Optional: You can add more tests here, like checking a batch
    print("\nTesting one batch from the train_loader...")
    data_batch, labels_batch = next(iter(train_loader))
    print(f"Data batch shape: {data_batch.shape}")
    print(f"Labels batch shape: {labels_batch.shape}")