import os
import sys

import hydra
import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf

from src.configs import TrainConfig
from src.model.model_module import ModelModule


@hydra.main(config_path="/kaggle/config", config_name="train", version_base="1.3")
def run_train(config: TrainConfig) -> None:
    print("Training!")
    print(f"Instantiating model: {config.model._target_}")
    model: ModelModule = hydra.utils.instantiate(config.model)

    # save weights
    os.makedirs("/kaggle/working/debug", exist_ok=True)
    model.save_state_dict(os.path.join("/kaggle", "working", "model.pth"))


if __name__ == "__main__":
    run_train()
    # print("run train")
