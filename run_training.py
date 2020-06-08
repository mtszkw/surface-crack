import os
import hydra
import torch
import neptune
import pytorch_lightning as pl
from omegaconf import DictConfig
from flatten_dict import flatten

from src.models import SurfaceCrackDetectionModel


@hydra.main(config_path="config.yaml")
def run_training(cfg : DictConfig) -> None:

    logger = pl.loggers.NeptuneLogger(
        api_key=None,
        project_name='mtszkw/surface-crack-detect',
        params=dict(cfg),
        tags=['binary-classification'],
        upload_source_files=['config.yaml']
    )

    model  = SurfaceCrackDetectionModel(hparams=dict(cfg))

    trainer = pl.Trainer(
        gpus=cfg.use_gpu,
        max_epochs=cfg.max_epochs,
        logger=logger,
        train_percent_check=0.2, val_percent_check=0.2, test_percent_check=0.2
    )

    trainer.fit(model)
    trainer.test()


if __name__ == '__main__':
    run_training()