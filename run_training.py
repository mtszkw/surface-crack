import os
import hydra
from omegaconf import DictConfig

import torch
import pytorch_lightning as pl

from src.models import SurfaceCrackDetectionModel


@hydra.main(config_path="config.yaml")
def run_training(cfg : DictConfig) -> None:
    print(cfg.pretty())

    tb_logger = pl.loggers.TensorBoardLogger(os.getcwd())
    model = SurfaceCrackDetectionModel(data_path=cfg.dataset.path)

    trainer = pl.Trainer(gpus=1, max_epochs=5, logger=tb_logger,
        train_percent_check=0.1, val_percent_check=0.1, test_percent_check=0.1
    )

    trainer.fit(model)
    trainer.test()


if __name__ == '__main__':
    run_training()