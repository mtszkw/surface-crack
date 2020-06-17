import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig

from src.LitModel import LitModel


@hydra.main(config_path="config.yaml")
def run_training(cfg : DictConfig) -> None:

    logger = pl.loggers.NeptuneLogger(
        api_key=None,
        params=dict(cfg),
        tags=['binary-classification'],
        project_name=cfg.project_name,
        experiment_name=cfg.experiment_name,
        offline_mode=cfg.neptune.offline_mode,
    )

    model  = LitModel(hparams=dict(cfg))

    lr_logger = pl.callbacks.LearningRateLogger()

    trainer = pl.Trainer(
        logger=logger,
        callbacks=[lr_logger],
        gpus=cfg.training.use_gpu,
        max_epochs=cfg.training.max_epochs,
        val_check_interval=cfg.training.val_check_interval,
        train_percent_check=cfg.debugging.train_percent_check,
        val_percent_check=cfg.debugging.val_percent_check,
        test_percent_check=cfg.debugging.test_percent_check
    )

    trainer.fit(model)
    trainer.test()


if __name__ == '__main__':
    run_training()