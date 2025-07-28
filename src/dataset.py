# src/dataset.py (Advanced, Config-Driven Version)

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, WeightedRandomSampler
import numpy as np
from . import config

def get_train_transforms():
    """Builds the training data augmentation pipeline from the config file."""
    aug_config = config.AUGMENTATION_CONFIG['train']
    
    train_transforms_list = [
        transforms.Resize((aug_config['resize_size'], aug_config['resize_size'])),
        transforms.RandomResizedCrop(aug_config['crop_size'], scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=aug_config['horizontal_flip_prob']),
        transforms.RandomRotation(aug_config['rotation_degrees']),
        transforms.RandomAffine(
            degrees=0, 
            translate=(aug_config['translate_range'], aug_config['translate_range']), 
            scale=aug_config['scale_range'], 
            shear=aug_config['shear_degrees']
        ),
        transforms.ColorJitter(
            brightness=aug_config['brightness'], 
            contrast=aug_config['contrast'], 
            saturation=aug_config['saturation']
        ),
    ]
    
    if aug_config.get('gaussian_blur_prob', 0) > 0:
        train_transforms_list.append(transforms.GaussianBlur(kernel_size=3, sigma=aug_config['gaussian_blur_sigma']))
        
    train_transforms_list.append(transforms.ToTensor())
    
    if aug_config.get('random_erasing_prob', 0) > 0:
        train_transforms_list.append(transforms.RandomErasing(p=aug_config['random_erasing_prob'], scale=aug_config['random_erasing_scale']))
        
    train_transforms_list.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    
    return transforms.Compose(train_transforms_list)

def get_val_test_transforms():
    """Builds the validation/test data transformation pipeline."""
    val_config = config.AUGMENTATION_CONFIG['val_test']
    
    return transforms.Compose([
        transforms.Resize((val_config['resize_size'], val_config['resize_size'])),
        transforms.CenterCrop(val_config['crop_size']),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def get_sampler(train_dataset):
    """Creates a weighted sampler based on the class balance configuration."""
    if not config.CLASS_BALANCE_CONFIG.get('use_weighted_sampling', False):
        return None

    train_targets = [s[1] for s in train_dataset.samples]
    class_counts = np.bincount(train_targets)
    
    class_weights = 1. / (class_counts + config.CLASS_BALANCE_CONFIG.get('smoothing_factor', 1e-6))
    class_weights = np.sqrt(class_weights) # Apply your sqrt logic
        
    sample_weights = class_weights[train_targets]
    
    return WeightedRandomSampler(
        weights=torch.DoubleTensor(sample_weights),
        num_samples=len(sample_weights),
        replacement=config.CLASS_BALANCE_CONFIG.get('replacement', True)
    )

def create_dataloaders():
    """Creates training and testing DataLoaders based on the config."""
    train_transforms = get_train_transforms()
    test_transforms = get_val_test_transforms()
    
    train_dataset = datasets.ImageFolder(config.TRAIN_DIR, transform=train_transforms)
    test_dataset = datasets.ImageFolder(config.TEST_DIR, transform=test_transforms)
    
    sampler = get_sampler(train_dataset)
    
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=config.BATCH_SIZE,
        sampler=sampler,
        shuffle=sampler is None,
        num_workers=config.TRAINING_CONFIG.get('num_workers', 4),
        pin_memory=config.HARDWARE_CONFIG.get('pin_memory', True),
        drop_last=True
    )
    
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.TRAINING_CONFIG.get('num_workers', 4),
        pin_memory=config.HARDWARE_CONFIG.get('pin_memory', True)
    )
    
    print("DataLoaders created successfully from advanced config.")
    return train_loader, test_loader, train_dataset, test_dataset