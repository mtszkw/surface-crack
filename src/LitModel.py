import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import pytorch_lightning as pl

import matplotlib.pyplot as plt
from omegaconf import DictConfig
from sklearn.metrics import f1_score, precision_recall_fscore_support

from src.AlexNet import alex_net
from src.DatasetProvider import read_dataset
from src.utils import make_weights_for_balanced_classes


class LitModel(pl.LightningModule):
    def __init__(self, hparams: DictConfig = None):
        super().__init__()
        self.hparams = hparams
        self.sigmoid = nn.Sigmoid()
        self.loss_fn = nn.BCELoss()
        self.model = alex_net(num_classes=1)


    def forward(self, x):
        x = self.model(x)
        return self.sigmoid(x)


    def prepare_data(self):
        self.train_ds, self.val_ds, self.test_ds = read_dataset(**self.hparams['dataset'])


    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters(), **self.hparams['optimizer'])
        scheduler = {
            'scheduler': lr_scheduler.StepLR(optimizer, **self.hparams['scheduler']),
            'name': 'learn_rate'
        }
        return [optimizer], [scheduler]


    @pl.data_loader
    def train_dataloader(self):
        # weights = make_weights_for_balanced_classes(self.train_ds.imgs, len(self.train_ds.classes))
        # weights = torch.DoubleTensor(weights)
        # sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))
        # train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size,shuffle = True,
            # sampler = sampler, num_workers=args.workers, pin_memory=True)

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
        prec, recall, f1, _ = precision_recall_fscore_support(y.cpu(), labels_hat.cpu(), average='binary')

        train_f1 = torch.tensor(f1)
        train_prec = torch.tensor(prec)
        train_recall = torch.tensor(recall)

        tensorboard_logs = {
            'train/loss': train_loss,
            'train/f1': train_f1,
            'train/prec': train_prec,
            'train/recall': train_recall
        }
        return {'loss': train_loss, 'f1': train_f1, 'log': tensorboard_logs}


    def training_epoch_end(self, outputs):
        train_epoch_loss = torch.stack([x['loss'] for x in outputs]).mean()
        return {'loss': train_epoch_loss}


    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        y_hat = self.forward(x)
        labels_hat = (y_hat > 0.5).int()

        val_loss = self.loss_fn(y_hat.flatten(), y.float())
        prec, recall, f1, _ = precision_recall_fscore_support(y.cpu(), labels_hat.cpu(), average='binary')

        val_f1 = torch.tensor(f1)
        val_prec = torch.tensor(prec)
        val_recall = torch.tensor(recall)

        return {'val_loss': val_loss, 'val_f1': val_f1, 'val_prec': val_prec, 'val_recall': val_recall}


    def validation_epoch_end(self, outputs):
        val_epoch_f1   = torch.stack([x['val_f1'] for x in outputs]).mean()
        val_epoch_prec = torch.stack([x['val_prec'] for x in outputs]).mean()
        val_epoch_recall = torch.stack([x['val_recall'] for x in outputs]).mean()
        val_epoch_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        
        tensorboard_logs = {
            'val/epoch_f1': val_epoch_f1,
            'val/epoch_prec': val_epoch_prec,
            'val/epoch_recall': val_epoch_recall,
            'val/epoch_loss': val_epoch_loss,
        }

        return {'val_loss': val_epoch_loss, 'log': tensorboard_logs}


    def test_step(self, test_batch, batch_idx):
        x, y = test_batch
        y_hat = self.forward(x)
        labels_hat = (y_hat > 0.5).int()

        test_loss = self.loss_fn(y_hat.flatten(), y.float())
        prec, recall, f1, _ = precision_recall_fscore_support(y.cpu(), labels_hat.cpu(), average='binary')

        test_f1 = torch.tensor(f1)
        test_prec = torch.tensor(prec)
        test_recall = torch.tensor(recall)

        tensorboard_logs = {    
            'test/loss': test_loss,
            'test/f1': test_f1,
            'test/prec': test_prec,
            'test/recall': test_recall
        }

        labels_dict = {0: 'Negative', 1: 'Positive'}
        labels_hat = labels_hat.flatten()
        wrong_ids_mask = torch.ne(labels_hat, y).nonzero().flatten().tolist()

        if len(wrong_ids_mask) > 0:
            for idx in wrong_ids_mask:
                img_name = f'test/pred-{labels_dict[labels_hat[idx].item()]}/true-{labels_dict[y[idx].item()]}/'
                self.logger.experiment.log_image(img_name,
                    torchvision.transforms.ToPILImage()(x[idx].cpu()).convert("RGB"))

        return {'test_loss': test_loss, 'test_f1': test_f1, 'test_recall': test_recall, 'test_prec': test_prec}
    

    def test_epoch_end(self, outputs):
        test_epoch_f1 = torch.stack([x['test_f1'] for x in outputs]).mean()
        test_epoch_prec = torch.stack([x['test_prec'] for x in outputs]).mean()
        test_epoch_recall = torch.stack([x['test_recall'] for x in outputs]).mean()
        test_epoch_loss = torch.stack([x['test_loss'] for x in outputs]).mean()

        self.logger.experiment.log_metric('test_epoch_f1', test_epoch_f1)
        self.logger.experiment.log_metric('test_epoch_prec', test_epoch_prec)
        self.logger.experiment.log_metric('test_epoch_recall', test_epoch_recall)
        self.logger.experiment.log_metric('test_epoch_loss', test_epoch_loss)

        return {
            'test_epoch_f1': test_epoch_f1,
            'test_epoch_prec': test_epoch_prec,
            'test_epoch_recall': test_epoch_recall,
            'test_epoch_loss': test_epoch_loss,
        }
