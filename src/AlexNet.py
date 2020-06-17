import torch
import torch.nn as nn
from torchvision import datasets, models, transforms

# Loads pre-trained model with frozen parameters and new classifier
def alex_net(num_classes, pretrained=True):
    model_ft = models.alexnet(pretrained=pretrained)
    
    for param in model_ft.parameters():
        param.requires_grad = False

    model_ft.classifier = nn.Sequential(
        nn.Dropout(),
        nn.Linear(256*6*6, 4096),
        nn.ReLU(inplace=True),
            
        nn.Dropout(),
        nn.Linear(4096, 4096),
        nn.ReLU(inplace=True),
            
        nn.Linear(4096, num_classes),
    )

    return model_ft