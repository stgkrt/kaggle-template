import os
import sys
from typing import Any

import pytorch_lightning as L
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))
from src.model.architectures.model_architectures import ModelArchitectures
from src.model.losses import LossModule


class ModelModule(L.LightningModule):
    def __init__(
        self,
        model_architectures: ModelArchitectures,
        loss: LossModule,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        compile: bool,
    ) -> None:
        super().__init__()
        self.model = model_architectures
        self.loss = loss

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.model(x)
        return x

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        x, y = batch
        pred = self.model(x)
        loss = self.loss(pred, y)
        self.log(
            "train_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        return loss

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        x, y = batch
        pred = self.model(x)
        loss = self.loss(pred, y)
        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        return loss

    def configure_optimizers(self) -> dict[str, Any]:  # type: ignore
        """Choose what optimizers and learning-rate schedulers
        to use in your optimization.
        Normally you'd need one.
        But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        Returns:
            A dict containing the configured optimizers and learning-rate schedulers
            to be used for training.
        """
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())  # type: ignore
        if self.hparams.scheduler is not None:  # type: ignore
            scheduler = self.hparams.scheduler(optimizer=optimizer)  # type: ignore
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}
