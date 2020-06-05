import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler

import pytorch_lightning as pl

from sklearn.metrics import f1_score

from utils import get_datasets, get_model


# SurfaceCrackDetectionModel
class SurfaceCrackDetectionModel(pl.LightningModule):
    def __init__(self, data_path):
        super().__init__()
        self.data_path = data_path
        self.model = get_model(num_classes=2)
    
    def forward(self, x):
        x = self.model(x)
        return x

    def prepare_data(self):
        self.train_dataset, self.val_dataset, self.test_dataset = get_datasets(self.data_path)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=16)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset, batch_size=16)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_dataset, batch_size=16)

    def configure_optimizers(self):
        optimizer = optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
        return [optimizer], [scheduler]

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        y_hat = self.forward(x)
        loss = nn.functional.cross_entropy(y_hat, y)
        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        y_hat = self.forward(x)
        loss = nn.functional.cross_entropy(y_hat, y)
        tensorboard_logs = {'val_loss': loss}
        return {'val_loss': loss, 'log': tensorboard_logs}

    def test_step(self, test_batch, batch_idx):
        x, y = test_batch
        y_hat = self.forward(x)
        loss = nn.functional.cross_entropy(y_hat, y)
        tensorboard_logs = {'test_loss': loss}
        return {'test_loss': loss, 'log': tensorboard_logs}

    def validation_epoch_end(self, outputs):
        val_loss_mean = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss_mean': val_loss_mean}
        return {'val_loss': val_loss_mean, 'log': tensorboard_logs}
    
    def test_epoch_end(self, outputs):
        test_loss_mean = torch.stack([x['test_loss'] for x in outputs]).mean()
        return {'test_loss': test_loss_mean}
