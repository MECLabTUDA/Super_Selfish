import torch
from torch import nn
from torch.nn import functional as F
from efficientnet_pytorch import EfficientNet
from utils import bcolors

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Library Modules
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


class ClassificationModule(nn.Module):
    def __init__(self, layers=[3136, 1024, 256, 4], activation=nn.PReLU, batchnorm=True):
        super().__init__()
        self.model = nn.Sequential(
            *[nn.Sequential(
                nn.Linear(in_features=layers[i - 1], out_features=layers[i]),
                nn.BatchNorm1d(
                    num_features=layers[i]) if batchnorm else nn.Identity(),
                activation())
              for i in range(1, len(layers))])

    def forward(self, x):
        return self.model(x)


class ReshapeFeatures(nn.Module):
    def __init__(self, backbone=None, in_channels=1280, out_channels=64, kernel_size=3, stride=1, padding=1, activation=nn.PReLU):
        super().__init__()
        if backbone is None:
            raise NotImplementedError(
                "You need to specify a backbone network.")
        self.backbone = backbone
        self.model = nn.Sequential(nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding), activation())

    def forward(self, x):
        x = self.backbone(x)
        return self.model(x).reshape(x.shape[0], -1)


class CombinedNet(nn.Module):
    def __init__(self, backbone=None, predictor=None):
        super().__init__()
        if backbone is None or predictor is None:
            raise NotImplementedError(
                "You need to specify a backbone and a predictor network.")
        self.backbone = backbone
        self.predictor = predictor
        self.model = nn.Sequential(self.backbone, self.predictor)

    def forward(self, x):
        return self.model(x)

    def save(self, name="store/base"):
        torch.save(self.model.state_dict(), name + ".pt")

    def load(self, name="store/base"):
        pretrained_dict = torch.load(name + ".pt")
        print("Loaded", name + ".pt")
        model_dict = self.model.state_dict()
        model_dict.update(pretrained_dict)
        self.model.load_state_dict(model_dict)

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Backbone Modules
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


class EfficientFeatures(nn.Module):
    def __init__(self, name='efficientnet-b0', pretrained=True):
        super().__init__()
        print(bcolors.OKBLUE, end="")
        if pretrained:
            self.model = EfficientNet.from_pretrained(name)
        else:
            self.model = EfficientNet.from_name(name)
        print(bcolors.ENDC, end="")

    def forward(self, x):
        return self.model.extract_features(x)
