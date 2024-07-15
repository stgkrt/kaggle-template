import hydra
import torch
from omegaconf import OmegaConf
from torchvision import datasets, transforms

from src.configs import TrainConfig


@hydra.main(config_path="/kaggle/config", config_name="train", version_base="1.3")
def run_train(config: TrainConfig) -> None:
    print("Training!")
    print(OmegaConf.to_yaml(config))
    print("\n")

    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.MNIST(
        "/kaggle/input/",  # データの保存先
        train=True,  # 学習用データを取得する
        download=True,  # データが無い時にダウンロードする
        transform=transform,  # テンソルへ
    )
    # # データローダー
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config.dataset.batch_size, shuffle=True
    )
    for inputs, labels in train_dataloader:
        print(inputs.shape, labels.shape)
        break




if __name__=='__main__':
    run_train()
