import torch
from torch import nn
from torch.nn import functional as F
from efficientnet_pytorch import EfficientNet
from utils import bcolors
import numpy as np

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Library Modules
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


class Classification(nn.Module):
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


class ChannelwiseFC(nn.Module):
    def __init__(self, backbone=None, layers=[64, 64, 64, 64], activation=nn.PReLU, batchnorm=True):
        super().__init__()
        if backbone is None:
            raise NotImplementedError(
                "You need to specify a backbone network.")
        self.backbone = backbone
        self.model = nn.Sequential(
            *[nn.Sequential(
                nn.Linear(in_features=layers[i - 1], out_features=layers[i]),
                nn.BatchNorm1d(
                    num_features=layers[i]) if batchnorm else nn.Identity(),
                activation())
              for i in range(1, len(layers))])

    def forward(self, x):
        x = self.backbone(x)
        x_shape = x.shape
        x = x.reshape(x_shape[0], x_shape[1], -1)
        x = x.reshape(-1, x.shape[2])
        return self.model(x).reshape(x_shape)


class Upsampling(nn.Module):
    def __init__(self, layers=[1280, 512, 256, 128, 3], kernel_size=5, stride=2, padding=2, activation=nn.ReLU, batchnorm=True):
        super().__init__()
        self.model = nn.Sequential(
            *[nn.Sequential(
                nn.ConvTranspose2d(
                    in_channels=layers[i - 1], out_channels=layers[i], kernel_size=kernel_size, stride=stride, padding=padding),
                nn.BatchNorm2d(
                    num_features=layers[i]) if batchnorm else nn.Identity(),
                activation())
              for i in range(1, len(layers))])

    def forward(self, x):
        return self.model(x)


class MaskedCNN(nn.Module):
    def __init__(self, layers=[1280, 512, 256, 128, 3],  mask=torch.ones((3, 3)).to('cuda'), stride=1, padding=1, activation=nn.PReLU, batchnorm=True):
        super().__init__()
        kernel_size = mask.shape[0]
        self.model = nn.Sequential(
            *[nn.Sequential(
                ConvMasked2d(mask=torch.unsqueeze(torch.unsqueeze(mask, dim=0), dim=0),
                             in_channels=layers[i -
                                                1], out_channels=layers[i], kernel_size=kernel_size, stride=stride, padding=padding),
                nn.BatchNorm2d(
                    num_features=layers[i]) if batchnorm else nn.Identity(),
                activation())
              for i in range(1, len(layers))])

    def forward(self, x):
        return self.model(x)


class GroupedUpsampling(nn.Module):
    def __init__(self, layers=[1280, 512, 256, 128], groups=np.array([50, 100]), kernel_size=5, stride=2, padding=2, activation=nn.ReLU, batchnorm=True):
        super().__init__()

        self.groups = groups
        self.c_per_group = layers[0]

        self.models = nn.ModuleList([nn.Sequential(
            *[nn.Sequential(
                nn.ConvTranspose2d(
                    in_channels=layers[i - 1], out_channels=layers[i], kernel_size=kernel_size, stride=stride, padding=padding),
                nn.BatchNorm2d(
                    num_features=layers[i]) if batchnorm else nn.Identity(),
                activation())
              for i in range(1, len(layers))],
            nn.ConvTranspose2d(
                in_channels=layers[-1], out_channels=groups[j], kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(
                num_features=groups[j]) if batchnorm else nn.Identity(),
            activation())
            for j in range(len(self.groups))])

    def forward(self, x):
        features = []
        for i in range(len(self.groups)):
            features.append(self.models[i](
                x[:, self.c_per_group * i: self.c_per_group * (i+1), :, :]))
        return torch.cat(features, dim=1)


class ReshapeChannels(nn.Module):
    def __init__(self, backbone=None, in_channels=1280, out_channels=64, kernel_size=3, stride=1, padding=1, activation=nn.PReLU, groups=1, flat=True):
        super().__init__()
        if backbone is None:
            raise NotImplementedError(
                "You need to specify a backbone network.")
        self.backbone = backbone
        self.model = nn.Sequential(nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding, groups=groups), activation())
        self.flat = flat

    def forward(self, x):
        x = self.backbone(x)
        if self.flat:
            return self.model(x).reshape(x.shape[0], -1)
        else:
            return self.model(x)


class GroupedCrossEntropyLoss(nn.Module):
    def __init__(self, loss=nn.CrossEntropyLoss, groups=np.array([50, 100]), reduction='mean'):
        super().__init__()

        self.groups = groups
        self.cum_groups = np.concatenate((
            np.zeros(1), np.cumsum(groups).astype(float))).astype(int)
        self.models = nn.ModuleList(
            [loss(reduction=reduction) for j in range(len(self.groups))])

    def forward(self, x, y):
        losses = 0
        for i in range(len(self.groups)):
            losses += (self.models[i](
                x[:, self.cum_groups[i]:self.cum_groups[i+1], :, :], y[:, i:(i+1), :, :].squeeze(1)))
        return losses


torch.autograd.set_detect_anomaly(True)


class InfoNCE(nn.Module):
    def __init__(self, target_shaper=ReshapeChannels(nn.Identity(), in_channels=1280, out_channels=64,
                                                     kernel_size=1, padding=0, activation=nn.Identity, flat=False),
                 k=2, ignore=3, N=2, reduction='mean'):
        super().__init__()
        self.k = k
        self.ignore = ignore
        self.N = N
        self.target_shaper = target_shaper
        self.reduction = reduction

    def forward(self, x, y):
        if self.target_shaper is not None:
            y = self.target_shaper(y)
        loss = 0
        encoding_size = y.shape[1]
        col_inds = torch.arange(x.shape[2]).long()
        for i in range(self.k):
            prediction = x[:, i *
                           encoding_size:(i+1) * encoding_size, :, :].clone()
            col_inds = (col_inds + 1) % x.shape[2]
            true_target = y[:, :, col_inds, :].clone()
            true_scalar = torch.exp(torch.sum(prediction * true_target, dim=1))

            false_scalar = true_scalar.clone()
            for j in range(self.N):
                false_target = y[torch.randperm(
                    x.shape[0]), :, :, :].clone()
                false_target = false_target[:, :,
                                            torch.randperm(x.shape[2]), :].clone()
                false_scalar += torch.exp(torch.sum(prediction *
                                                    false_target, dim=1))
            ratio = true_scalar / (false_scalar + 1e-7)
            ratio = ratio[:, self.ignore:(ratio.shape[1] - (i+1)), :]
            if self.reduction == 'mean':
                loss += torch.mean(ratio)
            elif self.reduction == 'sum':
                loss += torch.sum(ratio)
            else:
                raise NotImplementedError("Reduction type not implemented.")

        return loss


class Batch2Image(nn.Module):
    def __init__(self, backbone=None, new_shape=(8, 8)):
        # Expects BxCx1x1x1...., should be made more robust
        super().__init__()
        if backbone is None:
            raise NotImplementedError(
                "You need to specify a backbone network.")
        self.backbone = backbone
        self.new_shape = new_shape
        self.split = new_shape[0] * new_shape[1]

    def forward(self, x):
        x = self.backbone(x).squeeze()
        x = torch.stack(torch.split(x, self.split, dim=0))
        x = x.reshape(x.shape[0], self.new_shape[0],
                      self.new_shape[1], -1).permute(0, 3, 1, 2)
        return x


class CroppedSiamese(nn.Module):
    def __init__(self, backbone=None, half_crop_size=(25, 25)):
        super().__init__()
        if backbone is None:
            raise NotImplementedError(
                "You need to specify a backbone network.")
        self.backbone = backbone
        self.half_crop_size = half_crop_size

    def forward(self, x):
        # Way faster then sequential implementation
        n_x = x.shape[2] // self.half_crop_size[0]
        n_y = x.shape[3] // self.half_crop_size[1]

        crops = []
        for i in range(n_x):
            for j in range(n_y):
                crops.append(x[:, :, i * self.half_crop_size: (i+2) * self.half_crop_size,
                               j * self.half_crop_size: (j+2) * self.half_crop_size])

        crops = torch.cat(crops, dim=0)
        print(crops.shape)

        return self.model(crops)


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


class ConvMasked2d(nn.Conv2d):
    def __init__(self, mask, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.register_buffer('mask', mask)

    def forward(self, x):
        self.weight.data *= self.mask
        return super().forward(x)

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


class GroupedEfficientFeatures(nn.Module):
    def __init__(self, name='efficientnet-b0', pretrained=True, groups=np.array([1, 2])):
        super().__init__()
        print(bcolors.OKBLUE, end="")
        self.groups = groups
        self.cum_groups = np.concatenate((
            np.zeros(1), np.cumsum(groups).astype(float))).astype(int)
        if pretrained:
            self.expand = nn.ModuleList(
                [nn.Conv2d(groups[i], 3, 1) for i in range(len(self.groups))])
            self.models = nn.ModuleList(
                [EfficientNet.from_pretrained(name) for i in range(len(self.groups))])
        else:
            self.expand = nn.ModuleList(
                [nn.Conv2d(groups[i], 3, 1) for i in range(len(self.groups))])
            self.models = nn.ModuleList(
                [EfficientNet.from_name(name) for i in range(len(self.groups))])
        print(bcolors.ENDC, end="")

    def forward(self, x):
        features = []
        for i in range(len(self.groups)):
            features.append(self.models[i].extract_features(self.expand[i](
                x[:, self.cum_groups[i]:self.cum_groups[i+1], :, :])))
        return torch.cat(features, dim=1)
