import timm
import torch.nn as nn

from src.configs import ModelConfig


class BasicModel(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.model = timm.create_model(
            config.backbone_name,
            pretrained=config.pretrained,
            in_chans=config.in_channels,
            num_classes=config.out_channels,
        )

    def forward(self, x):
        x = self.model(x)
        return x


if __name__ == "__main__":
    config = ModelConfig(
        model_name="basic_model",
        backbone_name="resnet18",
        pretrained=True,
        in_channels=3,
        out_channels=1,
        use_batchnorm=True,
        dropout=0.5,
        loss_config=None,  # type: ignore
        optimizer=None,  # type: ignore
        scheduler=None,  # type: ignore
    )
    model = BasicModel(config)
    print(model)
