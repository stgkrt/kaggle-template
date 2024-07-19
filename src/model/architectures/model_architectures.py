import torch
from torch import nn
from torch.nn import functional as F

from src.configs import ModelConfig
from src.model.architectures.basic_model import BasicModel


class ModelArchitectures(nn.Module):
    def __init__(self, model_config: ModelConfig):
        super(ModelArchitectures, self).__init__()
        self.config = model_config
        self.model_name = model_config.model_name
        self.model = self._get_model()

    def _get_model(self):
        if self.model_name == "basic_model":
            model = BasicModel(self.config)
        else:
            raise NotImplementedError
        return model

    def forward(self, x):
        x = self.model(x)
        x = F.log_softmax(x, dim=1)
        return x


if __name__ == "__main__":
    config = ModelConfig(
        _target_="src.model.architectures.model_architectures.ModelArchitectures",
        model_name="basic_model",
        backbone_name="resnet18",
        pretrained=True,
        in_channels=1,
        out_channels=10,
        use_batchnorm=True,
        dropout=0.5,
        loss_config=None,  # type: ignore
        optimizer=None,  # type: ignore
        scheduler=None,  # type: ignore
    )
    model = ModelArchitectures(config)
    # model = ModelArchitectures()
    print(model)

    input = torch.randn(32, 1, 224, 224)
    output = model(input)
    print(output.shape)
