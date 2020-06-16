import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import pytorch_lightning as pl

import matplotlib.pyplot as plt
from omegaconf import DictConfig
from sklearn.metrics import f1_score
from scikitplot.metrics import plot_confusion_matrix

from src.utils import get_datasets, get_model


class SurfaceCrackDetectionModel(pl.LightningModule):
    def __init__(self, hparams: DictConfig = None):
        super().__init__()
        self.hparams = hparams
        self.sigmoid = nn.Sigmoid()
        self.loss_fn = nn.BCELoss()
        self.model = get_model(num_classes=1)

    def forward(self, x):
        x = self.model(x)
        return self.sigmoid(x)

    def prepare_data(self):
        self.train_ds, self.val_ds, self.test_ds = get_datasets(**self.hparams['dataset'])

    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters(), **self.hparams['optimizer'])
        scheduler = {
            'scheduler': lr_scheduler.StepLR(optimizer, **self.hparams['scheduler']),
            'name': 'learn_rate'
        }
        return [optimizer], [scheduler]

    @pl.data_loader
    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            dataset=self.train_ds,
            batch_size=self.hparams['training']['batch_size'],
            shuffle=True)

    @pl.data_loader
    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            dataset=self.val_ds,
            batch_size=self.hparams['training']['batch_size'])

    @pl.data_loader
    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            dataset=self.test_ds,
            batch_size=self.hparams['training']['batch_size'])

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        y_hat = self.forward(x)
        labels_hat = (y_hat > 0.5).int()

        train_loss = self.loss_fn(y_hat.flatten(), y.float())
        train_f1 = torch.tensor(f1_score(y.cpu(), labels_hat.cpu()))

        tensorboard_logs = {'train/loss': train_loss, 'train/f1': train_f1}
        return {'loss': train_loss, 'f1': train_f1, 'log': tensorboard_logs}

    def training_epoch_end(self, outputs):
        train_epoch_acc = torch.stack([x['f1'] for x in outputs]).mean()
        train_epoch_loss = torch.stack([x['loss'] for x in outputs]).mean()

        return {'loss': train_epoch_loss}

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        y_hat = self.forward(x)
        labels_hat = (y_hat > 0.5).int()

        val_loss = self.loss_fn(y_hat.flatten(), y.float())
        val_f1 = torch.tensor(f1_score(y.cpu(), labels_hat.cpu()))

        return {'val_loss': val_loss, 'val_f1': val_f1}

    def validation_epoch_end(self, outputs):
        val_epoch_f1   = torch.stack([x['val_f1'] for x in outputs]).mean()
        val_epoch_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        
        tensorboard_logs = {
            'val/epoch_f1': val_epoch_f1,
            'val/epoch_loss': val_epoch_loss,
        }

        return {'val_loss': val_epoch_loss, 'log': tensorboard_logs}

    def test_step(self, test_batch, batch_idx):
        x, y = test_batch
        y_hat = self.forward(x)
        labels_hat = (y_hat > 0.5).int()

        test_loss = self.loss_fn(y_hat.flatten(), y.float())
        test_f1   = torch.tensor(f1_score(y.cpu(), labels_hat.cpu()))
        
        labels_dict = {0: 'Negative', 1: 'Positive'}
        for idx, image in enumerate(x[labels_hat != y][:6]):
            img_name = 'img/pred-{}/true-{}/'.format(
                labels_dict[labels_hat[labels_hat != y].tolist()[idx]],
                labels_dict[y[labels_hat != y].tolist()[idx]])
            self.logger.experiment.log_image(
                img_name,
                torchvision.transforms.ToPILImage()(image.cpu()).convert("RGB"))

        return {'test_loss': test_loss, 'test_f1': test_f1}
    
    def test_epoch_end(self, outputs):
        test_epoch_f1 = torch.stack([x['test_f1'] for x in outputs]).mean()
        test_epoch_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        self.logger.experiment.log_metric('test_epoch_f1', test_epoch_f1)
        self.logger.experiment.log_metric('test_epoch_loss', test_epoch_loss)
        return {'test_epoch_loss': test_epoch_loss, 'test_epoch_f1': test_epoch_f1}
