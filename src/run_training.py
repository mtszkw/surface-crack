import torch
import pytorch_lightning as pl

from models import SurfaceCrackDetectionModel


def run_training():
    torch.cuda.empty_cache()
    print(f'CUDA {torch.version.cuda}')
    print(f'cuDNN {torch.backends.cudnn.version()}')

    model = SurfaceCrackDetectionModel(data_path='../../datasets/crack-detection')

    tb_logger = pl.loggers.TensorBoardLogger('logs/')
    trainer = pl.Trainer(
        gpus=1,
        max_epochs=5,
        logger=tb_logger,
        # progress_bar_refresh_rate=0,
        train_percent_check=0.1,
        val_percent_check=0.1,
        test_percent_check=0.1
    )

    trainer.fit(model)


if __name__ == '__main__':
    run_training()