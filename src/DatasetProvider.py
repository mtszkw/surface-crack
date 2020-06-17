import torch
import torch.nn as nn
from torchvision import datasets, models, transforms

class DatasetProvider:
    def __init__(self, path, training_pct, validation_pct):
        self.path = path
        self.training_pct = training_pct
        self.validation_pct = validation_pct

    def read(self):
        image_folder = datasets.ImageFolder(self.path)
        train_size = int(self.training_pct * len(image_folder))
        val_size   = int(self.validation_pct * len(image_folder))
        test_size  = len(image_folder) - (train_size + val_size)

        self.train_ds,
        self.val_ds,
        self.test_ds = torch.utils.data.random_split(image_folder, [train_size, val_size, test_size])

        print(f'Using {len(self.train_ds)} train., {len(self.val_ds)} valid. and {len(self.test_ds)} test samples.')

        num_positive = sum([x[1] for x in train_ds])
        print(f'Using {len(train_ds)} train samples, {num_positive} positive, {len(train_ds)-num_positive} negative.')

        train_ds.dataset.transform = transforms.Compose([
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

        return self.train_ds, self.val_ds, self.test_ds
