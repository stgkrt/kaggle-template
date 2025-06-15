import albumentations as A
import numpy as np
from albumentations.core.transforms_interface import BasicTransform
from hydra.utils import instantiate
from omegaconf import DictConfig, ListConfig

from src.log_utils import pylogger

log = pylogger.RankedLogger(__name__, rank_zero_only=True)


class Augmentations:
    def __init__(self, config: DictConfig):
        self.config = config
        self.augmentations = self._instantiate_transforms(config.aug_list)

    def __call__(self, image):
        # mnistに合わせて適当に変換
        image = (np.array(image) / 255.0).astype(np.float32)
        image = self.augmentations(image=image)["image"]
        return image

    def _instantiate_transforms(self, aug_list: list[BasicTransform]) -> A.Compose:
        """Instantiates transforms from config.
        Args:
            aug_list: A ListConfig object containing transforms.
        Returns:
            A.Compose: A composition of transforms.
        """
        transforms: list[BasicTransform] = []

        if not aug_list:
            log.warning("No transforms configs found! Skipping...")
            return transforms

        if not isinstance(aug_list, ListConfig):
            raise TypeError("auglist must be a list of augmentations!")

        for augmentation in aug_list:
            log.info(f"transform config = {augmentation}")
            if isinstance(augmentation, DictConfig):
                transforms.append(instantiate(augmentation))
            else:
                transforms.append(augmentation)

        return A.Compose(transforms)
