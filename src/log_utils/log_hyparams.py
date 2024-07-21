from typing import Any

from lightning_utilities.core.rank_zero import rank_zero_only
from omegaconf import OmegaConf

from src.log_utils import pylogger

log = pylogger.RankedLogger(__name__, rank_zero_only=True)


@rank_zero_only
def log_hyperparameters(object_dict: dict[str, Any]) -> None:
    """Controls which config parts are saved by Lightning loggers.

    Additionally saves:
        - Number of model parameters

    Args:
        object_dict: A dictionary containing the following objects:
            - `"cfg"`: A DictConfig object containing the main config.
            - `"model"`: The Lightning model.
            - `"trainer"`: The Lightning trainer.
    """
    hparams = {}

    cfg = OmegaConf.to_container(object_dict["cfg"])
    model = object_dict["model"]
    trainer = object_dict["trainer"]

    if not trainer.logger:
        log.warning("Logger not found! Skipping hyperparameter logging...")
        return

    hparams["model"] = cfg["model"]  # type: ignore

    # save number of model parameters
    hparams["model/params/total"] = sum(p.numel() for p in model.parameters())
    hparams["model/params/trainable"] = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    hparams["model/params/non_trainable"] = sum(
        p.numel() for p in model.parameters() if not p.requires_grad
    )

    hparams["data"] = cfg["data"]  # type: ignore
    hparams["trainer"] = cfg["trainer"]  # type: ignore

    hparams["callbacks"] = cfg.get("callbacks")  # type: ignore
    hparams["extras"] = cfg.get("extras")  # type: ignore

    hparams["task_name"] = cfg.get("task_name")  # type: ignore
    hparams["tags"] = cfg.get("tags")  # type: ignore
    hparams["ckpt_path"] = cfg.get("ckpt_path")  # type: ignore
    hparams["seed"] = cfg.get("seed")  # type: ignore

    # send hparams to all loggers
    for logger in trainer.loggers:
        logger.log_hyperparams(hparams)
