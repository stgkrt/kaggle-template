import os
import sys

import timm
import torch.nn as nn

model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir)
src_dir = os.path.join(model_dir, os.pardir)
sys.path.append(model_dir)
sys.path.append(src_dir)

from configs import ModelConfig


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
    )
    model = BasicModel(config)
    print(model)
