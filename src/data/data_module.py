import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from torchvision.datasets import MNIST
from torchvision.transforms import transforms

from src.configs import SplitConfig
from src.data.augmentations import Augmentations


class DataModule(LightningDataModule):
    """`LightningDataModule` for dataset.

    Read the docs:
        https://lightning.ai/docs/pytorch/latest/data/datamodule.html
    """

    def __init__(
        self,
        data_dir: str = "/kaggle/input",
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = True,
        splits: SplitConfig | None = None,
        train_transforms: Augmentations | None = None,
        valid_transforms: Augmentations | None = None,
    ) -> None:
        """Initialize a `MNISTDataModule`.

        Args:
            data_dir (str): The data directory. Defaults to `"data/"`.
            train_num (int): The number of training samples. Defaults to `55000`.
            valid_num (int): The number of validation samples. Defaults to `5000`.
            test_num (int): The number of test samples. Defaults to `10000`.
            batch_size (int): The batch size. Defaults to `64`.
            num_workers (int): The number of workers. Defaults to `0`.
            pin_memory (bool): Whether to pin memory. Defaults to `False`.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.data_train: Dataset | None = None
        self.data_val: Dataset | None = None
        self.data_test: Dataset | None = None
        # from config
        self.data_dir = data_dir
        if splits is None:
            train_val_test_split = (55000, 5000, 10000)
        else:
            train_val_test_split = (splits.train_num, splits.valid_num, splits.test_num)
        self.train_val_test_split = train_val_test_split
        self.batch_size = batch_size
        self.batch_size_per_device = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        # data transformations
        if train_transforms is None:
            self.train_transforms = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,)),
                    transforms.Resize((28, 28)),
                ]
            )
        else:
            self.train_transforms = train_transforms
        if valid_transforms is None:
            self.valid_transforms = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,)),
                    transforms.Resize((28, 28)),
                ]
            )
        else:
            self.valid_transforms = valid_transforms

    def train_dataloader(self) -> DataLoader:
        """Create and return the train dataloader.

        :return: The train dataloader.
        """
        return DataLoader(
            dataset=self.data_train,  # type: ignore
            batch_size=self.batch_size_per_device,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader:
        """Create and return the validation dataloader.

        :return: The validation dataloader.
        """
        return DataLoader(
            dataset=self.data_val,  # type: ignore
            batch_size=self.batch_size_per_device,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader:
        """Create and return the test dataloader.

        :return: The test dataloader.
        """
        return DataLoader(
            dataset=self.data_test,  # type: ignore
            batch_size=self.batch_size_per_device,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )

    def setup(self, stage: str | None = None) -> None:
        # Divide batch size by the number of devices.
        if self.trainer is not None:
            if self.batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.batch_size}) is not divisible"
                    + f"by the number of devices ({self.trainer.world_size})."
                )
            self.batch_size_per_device = self.batch_size // self.trainer.world_size

        # load and split datasets only if not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            trainset = MNIST(self.data_dir, train=True, transform=self.train_transforms)
            testset = MNIST(self.data_dir, train=False, transform=self.valid_transforms)
            dataset: Dataset = ConcatDataset(datasets=[trainset, testset])
            self.data_train, self.data_val, self.data_test = random_split(
                dataset=dataset,
                lengths=self.train_val_test_split,
                generator=torch.Generator().manual_seed(42),
            )


if __name__ == "__main__":
    from src.configs import DatasetConfig

    splits = SplitConfig(train_num=55000, valid_num=5000, test_num=10000)
    config = DatasetConfig(
        _target_="src.data.data_module.DataModule",
        data_dir="/kaggle/input",
        num_workers=0,
        batch_size=64,
        pin_memory=False,
        splits=splits,
        train_transforms=None,
        valid_transforms=None,
    )
    data_module = DataModule(
        data_dir=config.data_dir,
        num_workers=config.num_workers,
        batch_size=config.batch_size,
        pin_memory=config.pin_memory,
        splits=config.splits,
        train_transforms=None,
        valid_transforms=None,
    )
    data_module.setup()
    train_loader = data_module.train_dataloader()
    print(train_loader.dataset)
    for input, target in train_loader:
        print(input.shape, target.shape)
        break
