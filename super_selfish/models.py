import torch
from torch import nn
from torch.nn import functional as F
from .efficientnet_pytorch.model import EfficientNet
from .utils import bcolors
import numpy as np

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Library Modules
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

class View(nn.Module):
    def __init__(self, shape):
        """View as module for convenience

        Args:
            shape ([type]): New shape.
        """        
        super().__init__()
        self.shape = shape,  

    def forward(self, x):
        return x.view(*self.shape)

class Classification(nn.Module):
    def __init__(self, layers=[3136, 1024, 256, 4], activation=nn.PReLU, batchnorm=True):
        """Simple MLP.

        Args:
            layers (list, optional): Size of layers. Defaults to [3136, 1024, 256, 4].
            activation (torch.nn.Module identifier, optional): Activation function identifier. Defaults to nn.PReLU.
            batchnorm (bool, optional): Wether to use batchnorm. Defaults to True.
        """
        super().__init__()
        self.model = nn.Sequential(
            *[nn.Sequential(
                nn.Linear(in_features=layers[i - 1], out_features=layers[i]),
                nn.BatchNorm1d(
                    num_features=layers[i]) if batchnorm else nn.Identity(),
                activation())
              for i in range(1, len(layers))])

    def forward(self, x):
        if len(x.shape) == 4:
            x = x.view(x.shape[0], -1)
        return self.model(x)


class ChannelwiseFC(nn.Module):
    def __init__(self, layers=[49, 49, 49, 49], activation=nn.PReLU, batchnorm=True):
        """Channelwise FC MLP as used in context autoencoders.

        Args:
            backbone (torch.nn.Module, optional): Flat network to decorate. Defaults to None.
            layers (list, optional): Size of layers. Defaults to [49, 49, 49, 49].
            activation (torch.nn.Module identifier, optional): Activation function identifier. Defaults to nn.PReLU.
            batchnorm (bool, optional): Wether to use batchnorm. Defaults to True.

        Raises:
            NotImplementedError: Works only on top of another flat network.
        """
        super().__init__()
        self.model = nn.Sequential(
            *[nn.Sequential(
                nn.Linear(in_features=layers[i - 1], out_features=layers[i]),
                nn.BatchNorm1d(
                    num_features=layers[i]) if batchnorm else nn.Identity(),
                activation())
              for i in range(1, len(layers))])

    def forward(self, x):
        x_shape = x.shape
        x = x.reshape(x_shape[0], x_shape[1], -1)
        x = x.reshape(-1, x.shape[2])
        x = self.model(x).reshape(x_shape)
        return x


class Upsampling(nn.Module):
    def __init__(self, layers=[1280, 512, 256, 128, 3], kernel_size=4, stride=2, padding=1, activation=nn.PReLU, batchnorm=True, input_resolution=None, out_resolution=(225,225)):
        """Standard upconvolution network.

        Args:
            layers (list, optional): Size of layers. Defaults to [1280, 512, 256, 128, 3]
            kernel_size (int, optional): Size of kernel. Defaults to 5.
            stride (int, optional): Size of stride. Defaults to 2.
            padding (int, optional): Size of padding. Defaults to 2.
            activation (torch.nn.Module identifier, optional): Activation function identifier. Defaults to nn.PReLU.
            batchnorm (bool, optional): Wether to use batchnorm. Defaults to True.
            input_resolution ((int, int), optional): Reshape flat input. Set to None if not needed. Defaults to (7,7).
            out_resolution ((int, int), optional): Final resizing layer, may be handy for slight deviations. Set to None if not needed. Defaults to (225,225).
        """
        super().__init__()
        self.input_resolution = input_resolution
        self.out_resolution = out_resolution
        self.model = nn.Sequential(
            *[nn.Sequential(
                nn.ConvTranspose2d(
                    in_channels=layers[i - 1], out_channels=layers[i], kernel_size=kernel_size, stride=stride, padding=padding),
                nn.BatchNorm2d(
                    num_features=layers[i]) if batchnorm else nn.Identity(),
                activation())
              for i in range(1, len(layers))])

    def forward(self, x):
        if self.input_resolution is not None:
            x = x.view(x.shape[0], -1, self.input_resolution[0], self.input_resolution[1])
        x = self.model(x)
        if self.out_resolution is not None:
            x = F.interpolate(x, self.out_resolution)
        return x


class MaskedCNN(nn.Module):
    def __init__(self, layers=[1280, 512, 256, 128, 3],  mask=torch.ones((3, 3)).to('cuda'), stride=1,
                 padding=1, activation=nn.PReLU, batchnorm=True, side=['top']):
        """MaskedCNN as used for CPC v2

        Args:
            layers (list, optional): Size of layers. Defaults to [1280, 512, 256, 128, 3]
            mask (torch.FloatTensor, optional): Horizontal mask, rotation done in forward pass. Defaults to torch.ones((3, 3)).to('cuda').
            stride (int, optional): Size of stride. Defaults to 1.
            padding (int, optional): Size of padding. Defaults to 1.
            activation (torch.nn.Module identifier, optional): Activation function identifier. Defaults to nn.PReLU.
            batchnorm (bool, optional): Wether to use batchnorm. Defaults to True.
            side (list, optional): Which sides to use. Defaults to ['top'].
        """
        super().__init__()
        kernel_size = mask.shape[0]
        self.models = nn.Sequential(
            *[nn.Sequential(
                ConvMasked2d(mask=torch.unsqueeze(torch.unsqueeze(mask, dim=0), dim=0),
                             in_channels=layers[i -
                                                1], out_channels=layers[i], kernel_size=kernel_size, stride=stride, padding=padding),
                nn.BatchNorm2d(
                    num_features=layers[i]) if batchnorm else nn.Identity(),
                activation())
              for i in range(1, len(layers))])

        self.side = side

    def forward(self, x):
        if self.side == 'top':
            x_a = x
        elif self.side == 'bottom':
            x_a = torch.flip(x, dims=[2]).contiguous()
        elif self.side == 'right':
            x_a = torch.transpose(x, 2, 3).contiguous()
        elif self.side == 'left':
            x_a = torch.flip(torch.transpose(
                x, 2, 3).contiguous(), dims=[3]).contiguous()
        return self.models(x_a)


class GroupedUpsampling(nn.Module):
    def __init__(self, layers=[1280, 512, 256, 128], groups=np.array([50, 100]), kernel_size=4, stride=2, padding=1, activation=nn.ReLU, batchnorm=True, out_resolution=(225,225)):
        """Wraps grouping on upsampling.

        Args:
            layers (list, optional): Size of layers. Defaults to [1280, 512, 256, 128]
            groups ([int], optional): Number of channels per group. Defaults to np.array([50, 100]).
            kernel_size (int, optional): [description]. Defaults to 5.
            stride (int, optional): Size of stride. Defaults to 2.
            padding (int, optional): Size of padding. Defaults to 2.
            activation (torch.nn.Module identifier, optional): Activation function identifier. Defaults to nn.PReLU.
            batchnorm (bool, optional): Wether to use batchnorm. Defaults to True.
            out_resolution ((int, int), optional): Final resizing layer, may be handy for slight deviations. Set to None if not needed. Defaults to (225,225).
        """
        super().__init__()

        self.groups = groups
        self.c_per_group = layers[0]
        self.out_resolution = out_resolution
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
            features.append(F.interpolate(self.models[i](
                x[:, self.c_per_group * i: self.c_per_group * (i+1), :, :]), self.out_resolution))
        return torch.cat(features, dim=1)


class ReshapeChannels(nn.Module):
    def __init__(self, backbone=None, in_channels=1280, out_channels=64, kernel_size=3, stride=1, padding=1, activation=nn.PReLU, groups=1, flat=True):
        """Convolutional bottleneck (single layer)

        Args:
            backbone (torch.nn.Module, optional): ConvNet to decorate. Defaults to None.
            in_channels (int, optional): Number of input channels. Defaults to 1280.
            out_channels (int, optional): Number of output channels. Defaults to 64.
            kernel_size (int, optional): Size of kernel. Defaults to 3.
            stride (int, optional): Size of stride. Defaults to 1.
            padding (int, optional): Size of padding. Defaults to 1.
            activation (torch.nn.Module identifier, optional): Activation function identifier. Defaults to nn.PReLU.
            batchnorm (bool, optional): Wether to use batchnorm. Defaults to True.
            groups (int, optional): Number of equally sized groups. Defaults to 1.
            flat (bool, optional): Wether to flat output. Defaults to True.

        Raises:
            NotImplementedError: Works only on top of another ConvNet
        """
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


class GroupedLoss(nn.Module):
    def __init__(self, loss=nn.CrossEntropyLoss, groups=np.array([50, 100]), reduction='mean'):
        """Loss on groups of varying size.

        Args:
            loss (torch.nn.Module, optional): Criterion. Defaults to nn.CrossEntropyLoss.
            groups ([int], optional): Number of channels per group. Defaults to np.array([50, 100]).
            reduction (str, optional): Reduction that is supported by loss. Defaults to 'mean'.
        """
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


class CPCLoss(nn.Module):
    def __init__(self, target_shaper=ReshapeChannels(nn.Identity(), in_channels=64, out_channels=64,
                                                     kernel_size=1, padding=0, activation=nn.Identity, flat=False),
                 k=2, ignore=3, N=2, reduction='mean'):
        """Implements CPC v2 loss.

        Args:
            target_shaper ([torch.nn.Module], optional): Target head of CPC. Defaults to ReshapeChannels(nn.Identity(), in_channels=64, out_channels=64, kernel_size=1, padding=0, activation=nn.Identity, flat=False).
            k (int, optional): How many rows to predict. Defaults to 2.
            ignore (int, optional): How many rows to ignore. Defaults to 3.
            N (int, optional): Number of false targets. Defaults to 2.
            reduction (str, optional): Reduction that is supported by torch.nn.CrossEntropyLoss. Defaults to 'mean'.
        """
        super().__init__()
        self.k = k
        self.ignore = ignore
        self.N = N
        self.target_shaper = target_shaper
        self.loss = nn.CrossEntropyLoss(reduction=reduction)

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
            true_scalar = torch.sum(prediction * true_target, dim=1)

            full_scalar = [true_scalar]
            for j in range(self.N):
                false_target = y[torch.randperm(
                    x.shape[0]), :, :, :].clone()
                false_target = false_target[:, :,
                                            torch.randperm(x.shape[2]), :].clone()
                full_scalar.append(torch.sum(prediction *
                                             false_target, dim=1))
            full_scalar = torch.stack(full_scalar).permute(1, 0, 2, 3)
            full_scalar = full_scalar[:, :,
                                      self.ignore:(full_scalar.shape[2] - (i+1)), :]

            loss += self.loss(full_scalar,
                              torch.zeros(full_scalar.shape[0], full_scalar.shape[2], full_scalar.shape[3], device=full_scalar.device).long())

        return loss


class Batch2Image(nn.Module):
    def __init__(self, new_shape=(7, 7)):
        """ Concats batch of crops into one image.

        Args:
            new_shape (tuple, optional): Output image shape. Defaults to (7, 7).

        Raises:
            NotImplementedError: Works only on top of another ConvNet.
        """
        super().__init__()
        # Expects BxCx1x1x1...., should be made more robust
        self.new_shape = new_shape
        self.split = new_shape[0] * new_shape[1]

    def forward(self, x):
        x = x.squeeze()
        x = torch.stack(torch.split(x, self.split, dim=0))
        x = x.reshape(x.shape[0], self.new_shape[0],
                      self.new_shape[1], -1).permute(0, 3, 1, 2)
        return x


class CroppedSiamese(nn.Module):
    def __init__(self, backbone=None, half_crop_size=(28, 28)):
        """Deprecated.

        Args:
            backbone (torch.nn.Module, optional): ConvNet to decorate. Defaults to None.
            half_crop_size (tuple, optional): Half size of crops to predict. Defaults to (int(28), int(28)).

        Raises:
            NotImplementedError: Works only on top of another ConvNet.
        """
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

        return self.model(crops)


class SequentialUpTo(nn.Module):
    def __init__(self, *args, up_to=-1):
        """Sequential container like nn.Sequential that stops after a given layer (including)

        Args:
            up_to (int, optional): Layer to stop. Defaults to -1, no stopping.
        """
        super().__init__()
        self.ordered_models = nn.Sequential(*args)
        self.up_to = up_to

    def forward(self, x, up_to=-1):
        for i, model in enumerate(self.ordered_models):
            x = model(x)
            if i == up_to or i == self.up_to:
                break
        return x


class CombinedNet(nn.Module):
    def __init__(self, backbone=None, predictor=None, distributed=False):
        """Main building block of Super Selfish. Combines backbone features with a prediction head.
        Args:
            backbone (torch.nn.Module, optional): Backbone network. Defaults to None.
            predictor (torch.nn.Module, optional): Prediction network. Defaults to None.
            distributed (bool, optional): Wether to use nn.DataParallel, also handled in supervisor. Defaults to False.
        Raises:
            NotImplementedError: Backbone and Precitor must be specified.
        """
        super().__init__()
        if backbone is None or predictor is None:
            raise NotImplementedError(
                "You need to specify a backbone and a predictor network.")
        self.backbone = backbone
        self.predictor = predictor
        self.model = nn.DataParallel(nn.Sequential(self.backbone, self.predictor)) if distributed else nn.Sequential(self.backbone, self.predictor)

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
        """Masked convolution layer, same as torch.nn.Conv2d but mask.

        Args:
            mask ([float]): Mask of size kernel_size
        """
        super().__init__(*args, **kwargs)
        self.register_buffer('mask', mask)

    def forward(self, x):
        self.weight.data *= self.mask
        return super().forward(x)

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Backbone Modules
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


class EfficientFeatures(nn.Module):
    def __init__(self, name='efficientnet-b0', pretrained=False, norm_type='batch'):
        """Wrapper of EfficientNet features https://github.com/lukemelas/EfficientNet-PyTorch

        Args:
            name (str, optional): EfficientNet type. Defaults to 'efficientnet-b0'.
            pretrained (bool, optional): Wether to load pretrained weights. Defaults to False.
            norm_type (str, optional): ADDED FROM Super Selfish, decide between 'batch' and 'layer'. Defaults to 'batch'.
        """
        super().__init__()
        print(bcolors.OKBLUE, end="")
        if pretrained:
            self.model = EfficientNet.from_pretrained(
                name, norm_type=norm_type)
        else:
            self.model = EfficientNet.from_name(name, norm_type=norm_type)
        print(bcolors.ENDC, end="")

    def forward(self, x, endpoints=False):
        if endpoints:
            return self.model.extract_endpoints(x)
        return self.model.extract_features(x)


class GroupedEfficientFeatures(nn.Module):
    def __init__(self, name='efficientnet-b0', pretrained=False, groups=np.array([1, 2]), norm_type='batch', channels_per_group = 32):
        """Wrapps grouped EfficientNet with groups of varying size. Same parameters as EffcientFeatures but groups.
        Args:
            groups ([int], optional): Number of channels per group. Defaults to np.array([50, 100]).

        Args:
            name (str, optional): EfficientNet type. Defaults to 'efficientnet-b0'.
            pretrained (bool, optional): Wether to load pretrained weights. Defaults to False.
            norm_type (str, optional): ADDED FROM Super Selfish, decide between 'batch' and 'layer'. Defaults to 'batch'.
            groups ([type], optional): Group split. Defaults to np.array([1, 2]).
            channels_per_group (int, optional): Out channels per group. Defaults to 32.
        """
        super().__init__()
        print(bcolors.OKBLUE, end="")
        self.groups = groups
        channels_per_group = channels_per_group
        self.cum_groups = np.concatenate((
            np.zeros(1), np.cumsum(groups).astype(float))).astype(int)
        if pretrained:
            self.expand = nn.ModuleList(
                [nn.Conv2d(groups[i], 3, 1) for i in range(len(self.groups))])
            self.squeeze = nn.ModuleList(
                [nn.Conv2d(1280, channels_per_group, 1) for i in range(len(self.groups))])
            self.models = nn.ModuleList(
                [EfficientNet.from_pretrained(name, norm_type=norm_type) for i in range(len(self.groups))])
        else:
            self.expand = nn.ModuleList(
                [nn.Conv2d(groups[i], 3, 1) for i in range(len(self.groups))])
            self.squeeze = nn.ModuleList(
                [nn.Conv2d(1280, channels_per_group, 1) for i in range(len(self.groups))])
            self.models = nn.ModuleList(
                [EfficientNet.from_name(name, norm_type=norm_type) for i in range(len(self.groups))])
        print(bcolors.ENDC, end="")

    def forward(self, x):
        features = []
        for i in range(len(self.groups)):
            features.append(self.squeeze[i](self.models[i].extract_features(self.expand[i](
                x[:, self.cum_groups[i]:self.cum_groups[i+1], :, :]))))
        return torch.cat(features, dim=1)
