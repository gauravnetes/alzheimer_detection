import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from . import config

def create_enhanced_classifier(num_ftrs, num_classes, arch_config):
    """Dynamically creates an enhanced classifier from the config."""
    hidden_dims = arch_config['classifier_hidden_dims']
    dropout_rates = arch_config['classifier_dropout']
    use_batch_norm = arch_config['use_batch_norm']
    
    layers = []
    
    # Input layer
    if use_batch_norm:
        layers.append(nn.BatchNorm1d(num_ftrs))
    layers.append(nn.Dropout(dropout_rates[0]))
    layers.append(nn.Linear(num_ftrs, hidden_dims[0]))
    layers.append(nn.ReLU(inplace=True))
    
    # Hidden layers
    for i in range(len(hidden_dims) - 1):
        if use_batch_norm:
            layers.append(nn.BatchNorm1d(hidden_dims[i]))
        layers.append(nn.Dropout(dropout_rates[i+1]))
        layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
        layers.append(nn.ReLU(inplace=True))
        
    # Output layer
    if use_batch_norm:
        layers.append(nn.BatchNorm1d(hidden_dims[-1]))
    layers.append(nn.Dropout(dropout_rates[-1]))
    layers.append(nn.Linear(hidden_dims[-1], num_classes))
    
    return nn.Sequential(*layers)

def create_model(model_name=None):
    """
    Creates and fine-tunes a model based on the project configuration.
    """
    if model_name is None:
        model_name = config.TRAINING_CONFIG['model_name']
        
    arch_config = config.get_model_config(model_name)
    
    # --- Model Loading ---
    if model_name == 'resnet50':
        model = models.resnet50(weights=config.TRAINING_CONFIG['pretrained_weights'])
    elif model_name == 'efficientnet_b3':
        model = models.efficientnet_b3(weights=config.TRAINING_CONFIG['pretrained_weights'])
    elif model_name == 'densenet121':
        model = models.densenet121(weights=config.TRAINING_CONFIG['pretrained_weights'])
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    # --- Fine-Tuning Logic (Config-Driven) ---
    # Freeze all parameters initially
    for param in model.parameters():
        param.requires_grad = False
        
    # Unfreeze layers based on the config
    if model_name == 'resnet50':
        for layer_name in arch_config['unfreeze_layers']:
            layer = getattr(model, layer_name)
            for param in layer.parameters():
                param.requires_grad = True
    elif model_name == 'efficientnet_b3':
        num_blocks_to_unfreeze = arch_config['unfreeze_blocks']
        for param in model.features[-num_blocks_to_unfreeze:].parameters():
            param.requires_grad = True
    elif model_name == 'densenet121':
        for block_name in arch_config['unfreeze_blocks']:
            block = getattr(model.features, block_name)
            for param in block.parameters():
                param.requires_grad = True
                
    # --- Classifier Replacement (Dynamic) ---
    if model_name == 'resnet50':
        num_ftrs = model.fc.in_features
        model.fc = create_enhanced_classifier(num_ftrs, config.NUM_CLASSES, arch_config)
    elif model_name == 'efficientnet_b3':
        num_ftrs = model.classifier[1].in_features
        model.classifier = create_enhanced_classifier(num_ftrs, config.NUM_CLASSES, arch_config)
    elif model_name == 'densenet121':
        num_ftrs = model.classifier.in_features
        model.classifier = create_enhanced_classifier(num_ftrs, config.NUM_CLASSES, arch_config)
        
    print(f"Created {model_name} Architecture, fine-tuning driven by config.")
    return model.to(config.DEVICE)

class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance."""
    def __init__(self, alpha=1, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        return focal_loss.mean()

class EnsembleModel(nn.Module):
    """Ensemble of multiple models."""
    def __init__(self, models):
        super(EnsembleModel, self).__init__()
        self.models = nn.ModuleList(models)
        
    def forward(self, x):
        outputs = [model(x) for model in self.models]
        # Average the raw outputs (logits)
        ensemble_output = torch.stack(outputs).mean(dim=0)
        return ensemble_output

def create_ensemble_model(model_names=None):
    """Creates an ensemble of different architectures from the config."""
    if model_names is None:
        model_names = config.ENSEMBLE_CONFIG['models_to_train']
        
    trained_models = []
    for name in model_names:
        # Here you would typically load pre-trained models from disk
        # For demonstration, we create them. In a real scenario, you'd load weights.
        model = create_model(name)
        # model.load_state_dict(torch.load(f"path/to/best_{name}.pth"))
        trained_models.append(model)
        
    ensemble = EnsembleModel(trained_models)
    print(f"Created ensemble model with: {model_names}")
    return ensemble.to(config.DEVICE)