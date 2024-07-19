import os
import sys

import hydra
import pytorch_lightning as L
import torch

from src.configs import TrainConfig
from src.model.model_module import ModelModule


@hydra.main(config_path="/kaggle/config", config_name="train", version_base="1.3")
def run_train(config: TrainConfig) -> None:
    print("Training!")
    print(f"Instantiating model: {config.model._target_}")
    model: ModelModule = hydra.utils.instantiate(config.model)
    print(model)
    print(f"Instantiating data module: {config.dataset._target_}")
    datamodule = hydra.utils.instantiate(config.dataset)
    print(datamodule)
    print("Instantiating trainer")
    trainer: L.Trainer = hydra.utils.instantiate(config.trainer, logger=False)
    print(trainer)

    # object_dict = {
    #     "config": config,
    #     "model": model,
    #     "data_module": datamodule,
    #     "trainer": trainer,
    # }

    trainer.fit(model=model, datamodule=datamodule)

    # save weights
    # os.makedirs("/kaggle/working/debug", exist_ok=True)
    # model.save_state_dict(os.path.join("/kaggle", "working", "model.pth"))


if __name__ == "__main__":
    run_train()
    # print("run train")
