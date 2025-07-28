import torch

# Project settings 
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TRAIN_DIR = "Alzheimer_Dataset/data/train"
TEST_DIR = "Alzheimer_Dataset/data/test"
MODEL_SAVE_PATH = "Alzheimer_Dataset/saved_models/"

# Dynamic run naming based on model and configuration
RUN_NAME = "EfficientNet_FocalLoss_Mixup"  # More descriptive naming

# Model Hyperparameters - Optimized for better performance
NUM_CLASSES = 4
BATCH_SIZE = 16  # Reduced for better gradient estimates and memory efficiency
NUM_EPOCHS = 25  # Increased for better convergence
LEARNING_RATE = 0.0001  # Base LR (will be adjusted per layer in training)
RANDOM_SEED = 42

# Enhanced training configurations
TRAINING_CONFIG = {
    # Model selection
    'model_name': 'efficientnet_b3',  # Better than ResNet50 for medical images
    'pretrained_weights': 'IMAGENET1K_V1',
    
    # Loss function settings
    'use_focal_loss': True,
    'focal_alpha': 1.0,
    'focal_gamma': 2.0,
    'label_smoothing': 0.1,
    
    # Augmentation settings
    'use_mixup': True,
    'mixup_alpha': 0.2,
    'mixup_prob': 0.5,
    
    # Optimizer settings
    'optimizer': 'adamw',
    'weight_decay': 1e-4,
    'gradient_clip_norm': 1.0,
    
    # Learning rate settings
    'base_lr': LEARNING_RATE,
    'lr_scheduler': 'cosine_warm_restarts',
    'warmup_epochs': 3,
    'min_lr': 1e-7,
    
    # Layer-specific learning rates (multipliers of base_lr)
    'lr_multipliers': {
        'backbone_early': 0.01,    # Early layers: very small LR
        'backbone_mid': 0.05,      # Middle layers: small LR  
        'backbone_late': 0.1,      # Late layers: moderate LR
        'classifier': 10.0         # New classifier: high LR
    },
    
    # Early stopping and regularization
    'early_stopping_patience': 10,
    'early_stopping_min_delta': 0.001,
    'dropout_rate': 0.5,
    
    # Training monitoring
    'save_best_only': True,
    'monitor_metric': 'val_accuracy',
    'save_training_plots': True,
    
    # Data loading
    'num_workers': 4,
    'pin_memory': True,
    'persistent_workers': True,
}

# Evaluation configurations
EVALUATION_CONFIG = {
    'use_tta': True,
    'tta_num_augments': 5,
    'confidence_threshold': 0.9,
    'save_detailed_reports': True,
}

# Ensemble configurations
ENSEMBLE_CONFIG = {
    'models_to_train': ['resnet50', 'efficientnet_b3', 'densenet121'],
    'ensemble_method': 'average',  # 'average', 'weighted_average', 'voting'
    'weights': [0.3, 0.5, 0.2],  # Weights for weighted average (if used)
}

# Data augmentation configurations
AUGMENTATION_CONFIG = {
    'train': {
        'resize_size': 256,
        'crop_size': 224,
        'horizontal_flip_prob': 0.5,
        'rotation_degrees': 15,
        'translate_range': 0.05,
        'scale_range': (0.95, 1.05),
        'shear_degrees': 5,
        'brightness': 0.2,
        'contrast': 0.2,
        'saturation': 0.1,
        'gaussian_blur_prob': 0.1,
        'gaussian_blur_sigma': (0.1, 1.0),
        'random_erasing_prob': 0.1,
        'random_erasing_scale': (0.02, 0.1),
    },
    'val_test': {
        'resize_size': 256,
        'crop_size': 224,
        'center_crop': True,
    }
}

# Class imbalance handling
CLASS_BALANCE_CONFIG = {
    'use_weighted_sampling': True,
    'sampling_strategy': 'sqrt_inv_freq',  # 'inverse_freq', 'sqrt_inv_freq', 'log_inv_freq'
    'smoothing_factor': 0.1,
    'replacement': True,
}

# Model architecture configurations
ARCHITECTURE_CONFIG = {
    'resnet50': {
        'unfreeze_layers': ['layer2', 'layer3', 'layer4'],
        'classifier_hidden_dims': [1024, 512],
        'classifier_dropout': [0.3, 0.5, 0.4],
        'use_batch_norm': True,
    },
    'efficientnet_b3': {
        'unfreeze_blocks': 3,  # Unfreeze last 3 blocks
        'classifier_hidden_dims': [1024, 512],  
        'classifier_dropout': [0.3, 0.5, 0.4],
        'use_batch_norm': True,
    },
    'densenet121': {
        'unfreeze_blocks': ['denseblock3', 'denseblock4'],
        'classifier_hidden_dims': [1024, 512],
        'classifier_dropout': [0.3, 0.5, 0.4],
        'use_batch_norm': True,
    }
}

# Logging and monitoring
LOGGING_CONFIG = {
    'log_level': 'INFO',
    'log_file': 'training.log',
    'tensorboard_log_dir': 'runs/',
    'save_model_every_n_epochs': 5,
    'print_freq': 100,  # Print training stats every N batches
    'plot_training_curves': True,
}

# Hardware optimization
HARDWARE_CONFIG = {
    'mixed_precision': torch.cuda.is_available(),  # Use AMP if CUDA available
    'compile_model': False,  # Set to True if using PyTorch 2.0+
    'channels_last': False,   # Memory format optimization
}

# Cross-validation settings (if needed)
CV_CONFIG = {
    'use_cross_validation': False,
    'n_folds': 5,
    'stratified': True,
    'random_state': RANDOM_SEED,
}

# Medical imaging specific settings
MEDICAL_CONFIG = {
    'normalize_intensity': True,
    'clip_intensity_range': (-1000, 1000),  # HU range for CT, adjust for MRI
    'apply_window_level': False,
    'skull_strip': False,  # Set to True if skull stripping is needed
}

def get_model_config(model_name):
    """Get configuration for specific model"""
    return ARCHITECTURE_CONFIG.get(model_name, ARCHITECTURE_CONFIG['efficientnet_b3'])

def get_lr_for_layer_group(layer_group):
    """Get learning rate for specific layer group"""
    base_lr = TRAINING_CONFIG['base_lr']
    multiplier = TRAINING_CONFIG['lr_multipliers'].get(layer_group, 1.0)
    return base_lr * multiplier

def print_config():
    """Print current configuration"""
    print("="*60)
    print("CURRENT CONFIGURATION")
    print("="*60)
    print(f"Device: {DEVICE}")
    print(f"Model: {TRAINING_CONFIG['model_name']}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Epochs: {NUM_EPOCHS}")
    print(f"Base Learning Rate: {LEARNING_RATE}")
    print(f"Use Focal Loss: {TRAINING_CONFIG['use_focal_loss']}")
    print(f"Use Mixup: {TRAINING_CONFIG['use_mixup']}")
    print(f"Use TTA: {EVALUATION_CONFIG['use_tta']}")
    print("="*60)

# Validation
assert NUM_CLASSES == 4, "This config is optimized for 4-class Alzheimer's classification"
assert BATCH_SIZE > 0, "Batch size must be positive"
assert NUM_EPOCHS > 0, "Number of epochs must be positive"

# Auto-adjust batch size based on available memory (optional)
def auto_adjust_batch_size():
    """Automatically adjust batch size based on GPU memory"""
    if torch.cuda.is_available():
        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        
        if gpu_memory_gb >= 24:  # RTX 4090, A100, etc.
            return 32
        elif gpu_memory_gb >= 12:  # RTX 4070 Ti, RTX 3060, etc.
            return 24
        elif gpu_memory_gb >= 8:   # RTX 3060, GTX 1070, etc.
            return 16
        else:                      # Lower memory GPUs
            return 8
    return 16  # Default for CPU

# Uncomment the following line to auto-adjust batch size
# BATCH_SIZE = auto_adjust_batch_size()