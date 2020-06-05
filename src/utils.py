import torch
import torch.nn as nn
from torchvision import datasets, models, transforms

# Loads images and returns three separate datasets (training, validation and test)
def get_datasets(data_path):
    image_folder = datasets.ImageFolder(data_path)
    train_size = int(0.5 * len(image_folder))
    valid_size = int(0.3 * len(image_folder))
    test_size  = len(image_folder) - (train_size + valid_size)
    train_ds, val_ds, test_ds = torch.utils.data.random_split(image_folder, [train_size, valid_size, test_size])

    print(f'There are {len(train_ds)} train., {len(val_ds)} valid. and {len(test_ds)} test samples.')

    train_ds.dataset.transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    val_ds.dataset.transform =  transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    test_ds.dataset.transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    return train_ds, val_ds, test_ds


# Loads pre-trained VGG16 model with frozen parameters and new FC layer.
def get_model(num_classes):
    model = models.vgg16(pretrained=True)
        
    for param in model.parameters():
        param.requires_grad = False
        
    model.classifier[6] = nn.Linear(4096, num_classes)
    return model