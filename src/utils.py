import torch
import torch.nn as nn
from torchvision import datasets, models, transforms

# Loads images and returns three separate datasets (training, validation and test)
def get_datasets(path, training_pct, validation_pct):
    image_folder = datasets.ImageFolder(path)
    train_size = int(training_pct * len(image_folder))
    val_size   = int(validation_pct * len(image_folder))
    test_size  = len(image_folder) - (train_size + val_size)
    train_ds, val_ds, test_ds = torch.utils.data.random_split(image_folder, [train_size, val_size, test_size])

    print(f'There are {len(train_ds)} train., {len(val_ds)} valid. and {len(test_ds)} test samples.')

    # Training
    num_positive = sum([x[1] for x in train_ds])
    print(f'Using {len(train_ds)} train samples, {num_positive} positive, {len(train_ds)-num_positive} negative.')

    train_ds.dataset.transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Validation
    num_positive = sum([x[1] for x in val_ds])
    print(f'Using {len(val_ds)} val. samples, {num_positive} positive, {len(val_ds)-num_positive} negative.')

    val_ds.dataset.transform =  transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Test
    num_positive = sum([x[1] for x in test_ds])
    print(f'Using {len(test_ds)} test samples, {num_positive} positive, {len(test_ds)-num_positive} negative.')

    test_ds.dataset.transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    return train_ds, val_ds, test_ds


# Loads pre-trained VGG16 model with frozen parameters and new FC layer.
def get_model(num_classes):
    model_ft = models.vgg16(pretrained=True)
    # model_ft = models.alexnet(pretrained=True)
    
    # model_ft.classifier  = nn.Sequential(
        # nn.Dropout(0.3),
        # nn.Linear(256 * 6 * 6, 4096),
        # nn.ReLU(inplace=True),
        # nn.Dropout(0.3),
        # nn.Linear(4096, 4096),
        # nn.ReLU(inplace=True),
        # nn.Linear(4096, num_classes),
    # )

    # num_ftrs = model_ft.fc.in_features
    # model_ft.fc = nn.Linear(num_ftrs, len(num_classes))
        
    for param in model_ft.parameters():
        param.requires_grad = False
        
    model_ft.classifier[6] = nn.Linear(4096, num_classes)

    return model_ft