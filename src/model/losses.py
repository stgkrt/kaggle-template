import os
import sys

import torch
from torch import nn

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))
from configs import LossConfig


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        # comment out if your model contains a sigmoid or equivalent activation layer
        inputs = inputs.sigmoid()

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice = (2.0 * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
        if torch.isnan(dice):
            print("input", torch.isnan(inputs))
            print("target", torch.isnan(targets))
            print("intersection", torch.isnan(intersection))
            raise RuntimeError

        return 1 - dice


class LossModule(nn.Module):
    def __init__(self, loss_config: LossConfig):
        super(LossModule, self).__init__()
        self.loss_name = loss_config.loss_name
        self.pos_weight = loss_config.pos_weight
        self.loss = self._set_loss()

    def forward(self, inputs, targets):
        return self.loss(inputs, targets)

    def _set_loss(self) -> nn.Module:
        print("loss name", self.loss_name)
        if self.loss_name == "BCEWithLogitsLoss":
            if self.pos_weight is None:
                print("pos weight is None")
                loss: nn.Module = nn.BCEWithLogitsLoss()
            else:
                print("pos weight is not None", self.pos_weight)
                pos_weight = torch.tensor(self.pos_weight)
                loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        elif self.loss_name == "CrossEntropyLoss":
            loss = nn.BCELoss()
        elif self.loss_name == "DiceLoss":
            loss = DiceLoss()
        else:
            raise NotImplementedError
        return loss


if __name__ == "__main__":
    config = LossConfig(
        loss_name="BCEWithLogitsLoss",
        pos_weight=None,
    )
    loss = LossModule(config)
    pred = torch.randn(4, 1, 256, 256)
    target = torch.randn(4, 1, 256, 256)

    print(loss(pred, target))
