from typing import Any

import hydra
import pytorch_lightning as L
from omegaconf import DictConfig, OmegaConf

from src.log_utils.pylogger import RankedLogger
from src.log_utils.tasks import extras, task_wrapper
from src.model.model_module import ModelModule

log = RankedLogger(__name__, rank_zero_only=True)


@hydra.main(config_path="/kaggle/config", config_name="train", version_base="1.3")
def main(config: DictConfig) -> None:
    extras(config)
    metrics_dict, _ = run_train(config)
    print("metrics", metrics_dict)


@task_wrapper
def run_train(config: DictConfig) -> tuple[dict[str, Any], dict[str, Any]]:
    log.info(f"{__name__} started with config: \n{OmegaConf.to_yaml(config)}")
    if config.get("seed"):
        log.info(f"Setting seed: {config.seed}")
        L.seed_everything(config.seed)

    log.info(f"Instantiating model: {config.model._target_}")
    model: ModelModule = hydra.utils.instantiate(config.model)
    log.info(f"Instantiating data module: {config.dataset._target_}")
    datamodule = hydra.utils.instantiate(config.dataset)
    log.info("Instantiating trainer")
    trainer: L.Trainer = hydra.utils.instantiate(config.trainer, logger=False)
    log.info("Starting training")
    trainer.fit(model=model, datamodule=datamodule)
    train_metrics = trainer.callback_metrics

    # log用のdictを作成。task_wrapperで返す値は、(metrics_dict, object_dict)のタプル
    object_dict = {
        "cfg": config,
        "model": model,
        "trainer": trainer,
    }
    metrics_dict = {**train_metrics}
    return metrics_dict, object_dict


if __name__ == "__main__":
    main()
