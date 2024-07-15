import hydra
import pytorch_lightning as pl
import torch
from configs import TrainConfig
from model.model_module import ModelModule
from omegaconf import OmegaConf


@hydra.main(config_path="/kaggle/config", config_name="train", version_base="1.3")
def run_train(config: TrainConfig) -> None:
    print("Training!")
    print(OmegaConf.to_yaml(config))
    print("\n")
    print(OmegaConf.to_yaml(config.model))
    print(f"Instantiating model: {config.model._target_}")
    model: ModelModule = hydra.utils.instantiate(config.model)
    print(model)


if __name__ == "__main__":
    run_train()
