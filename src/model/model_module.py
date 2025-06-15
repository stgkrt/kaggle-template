import os
from typing import Any

import polars as pl
import pytorch_lightning as L
import torch
from torchmetrics import MeanMetric

from src.metrics.competition_metrics import CompetitionMetrics
from src.model.architectures.model_architectures import ModelArchitectures
from src.model.losses import LossModule


class ModelModule(L.LightningModule):
    def __init__(
        self,
        model_architectures: ModelArchitectures,
        criterion: LossModule,
        metrics: CompetitionMetrics,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        compile: bool,
        oof_dir: str,
    ) -> None:
        super().__init__()
        self.model = model_architectures
        self.criterion = criterion
        self.metrics = metrics
        self.best_metrics = -torch.inf
        self.oof_dir = oof_dir
        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.train_loss = MeanMetric()
        self.valid_loss = MeanMetric()
        # validの予測値と正解値を保存するための変数(device設定)
        if torch.cuda.is_available():
            self.accelarator = "cuda"
        else:
            self.accelarator = "cpu"
        self.valid_preds = torch.Tensor().to(self.accelarator)
        self.valid_targets = torch.Tensor().to(self.accelarator)

    def setup(self, stage: str) -> None:
        if self.hparams.compile and stage == "fit":  # type: ignore
            self.model = torch.compile(self.model)  # type: ignore

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.model(x)
        return x

    def model_step(
        self, batch: tuple[torch.Tensor, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        inputs, targets = batch
        logits = self.forward(inputs)
        # targetを0~9の値から各クラスの確率に変換
        targets_oe = torch.nn.functional.one_hot(targets, num_classes=10)
        targets_oe = targets_oe.float()
        loss = self.criterion(logits, targets_oe)
        preds = torch.argmax(logits, dim=1)
        return loss, preds, targets

    def training_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        loss, preds, targets = self.model_step(batch)
        self.train_loss(loss)
        self.log(
            "train_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        return loss

    def validation_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        loss, preds, targets = self.model_step(batch)
        self.valid_preds = torch.cat((self.valid_preds, preds))
        self.valid_targets = torch.cat((self.valid_targets, targets))
        self.valid_loss(loss)
        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        return loss

    # epochの終わりにmetricsのlogを出力
    def on_train_epoch_end(self) -> None:
        valid_targets = self.valid_targets.cpu()
        valid_preds = self.valid_preds.cpu()
        metrics = self.metrics(valid_targets, valid_preds)
        self.log(
            "competition_metrics", metrics, on_step=False, on_epoch=True, prog_bar=True
        )
        self.save_best(metrics)
        # preds/targetsの初期化
        self.valid_preds = torch.Tensor().to(self.accelarator)
        self.valid_targets = torch.Tensor().to(self.accelarator)
        return super().on_train_epoch_end()

    def save_best(self, metrics: float) -> None:
        if metrics > self.best_metrics:
            self.best_metrics = metrics
            # oofの保存
            oof = pl.DataFrame(
                {
                    "target": self.valid_targets.cpu().numpy(),
                    "preds": self.valid_preds.cpu().numpy(),
                }
            )
            oof.write_csv(os.path.join(self.oof_dir, "oof.csv"))
            # load best weights
            weights_path = os.path.join(self.oof_dir, "best_weights.pth")
            torch.save(self.model.model.state_dict(), weights_path)

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

    def save_state_dict(self, path: str) -> None:
        torch.save(self.model.state_dict(), path)
        print(f"Model weights saved to {path}")
        return
