import torch.nn as nn
from torchvision import models
from . import config

def create_model(): 
    # defining the ResNet50 model for transfer learning
    model = models.resnet50(weights='IMAGENET1K_V1')
    
    for param in model.parameters(): 
        param.requires_grad = False
        
    for param in model.layer4.parameters(): 
        param.requires_grad = True
        
    for param in model.layer3.parameters():
        param.requires_grad = True
    
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 512), 
        nn.ReLU(), 
        nn.Dropout(0.6), # regularization -> randomly sets 50% of neuron activations to zero during training to prevent overfitting. 
        nn.Linear(512, config.NUM_CLASSES)
    )
    
    print("Created Model Architecture [Fine-tuning enabled for layer3, layer4 and fc]")
    return model.to(config.DEVICE)