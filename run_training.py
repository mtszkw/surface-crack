import os
import hydra
import torch
import neptune
import pytorch_lightning as pl

from omegaconf import DictConfig
from flatten_dict import flatten
from src.SurfaceCrackDetectionModel import SurfaceCrackDetectionModel


@hydra.main(config_path="config.yaml")
def run_training(cfg : DictConfig) -> None:

    logger = pl.loggers.NeptuneLogger(
        api_key=None,
        offline_mode=cfg.neptune.offline_mode,
        project_name='mtszkw/surface-crack-detect',
        experiment_name='Transfer',
        params=dict(cfg),
        tags=['binary-classification'],
        upload_source_files=['*.yaml']
    )

    model  = SurfaceCrackDetectionModel(hparams=dict(cfg))

    lr_logger = pl.callbacks.LearningRateLogger()

    trainer = pl.Trainer(
        gpus=cfg.use_gpu,
        max_epochs=cfg.max_epochs,
        logger=logger,
        callbacks=[lr_logger],
        # weights_summary=None,
        train_percent_check=cfg.training.train_percent_check,
        val_percent_check=cfg.training.val_percent_check,
        test_percent_check=cfg.training.test_percent_check
    )

    trainer.fit(model)
    trainer.test()


if __name__ == '__main__':
    run_training()