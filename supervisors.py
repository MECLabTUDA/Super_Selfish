import torch
from torch import nn
from torch.nn import functional as F
from data import visualize, RotateDataset, ExemplarDataset, JigsawDataset, DenoiseDataset, ContextDataset, BiDataset, SplitBrainDataset, ContrastivePreditiveCodingDataset
from models import ReshapeChannels, Classification, Batch2Image, GroupedCrossEntropyLoss, InfoNCE, MaskedCNN, EfficientFeatures, CombinedNet, Upsampling, ChannelwiseFC, GroupedEfficientFeatures, GroupedUpsampling
from tqdm import tqdm
from colorama import Fore
from utils import bcolors
import numpy as np


class Supervisor():
    def __init__(self, model, dataset, loss=nn.CrossEntropyLoss(reduction='mean')):
        self.model = model
        self.dataset = dataset
        self.loss = loss

    def supervise(self, lr=1e-3, optimizer=torch.optim.Adam, epochs=10, batch_size=32, shuffle=True,
                  num_workers=0, name="store/base", pretrained=False, collate_fn=None):
        print(bcolors.OKGREEN + "Train with " +
              type(self).__name__ + bcolors.ENDC)
        try:
            if pretrained:
                self.load(name)
        except Exception:
            raise IOError("Could not load pretrained.")
        try:
            train_loader = torch.utils.data.DataLoader(self.dataset, batch_size=batch_size,
                                                       shuffle=shuffle, num_workers=num_workers, collate_fn=collate_fn)
            optimizer = optimizer(self.model.parameters(), lr=lr)
            for epoch_id in range(epochs):
                loss_sum = 0
                tkb = tqdm(total=int(len(train_loader)), bar_format="{l_bar}%s{bar}%s{r_bar}" % (
                    Fore.GREEN, Fore.RESET), desc="Batch Process Epoch " + str(epoch_id))

                for batch_id, data in enumerate(train_loader):
                    inputs, labels = data
                    optimizer.zero_grad()
                    outputs = self.model(inputs.to('cuda'))
                    loss = self.loss(outputs, labels.to('cuda'))
                    loss.backward()
                    optimizer.step()
                    loss_sum += loss.item()
                    tkb.set_postfix(loss='{:3f}'.format(
                        loss_sum / (batch_id+1)))
                    tkb.update(1)
        finally:
            self.save(name)
            print()

    def to(self, name):
        self.model = self.model.to(name)
        return self

    def get_backbone(self):
        return self.model.backbone

    def save(self, name="store/base"):
        torch.save(self.model.state_dict(), name + ".pt")
        print(bcolors.OKBLUE + "Saved at " + name + "." + bcolors.ENDC)

    def load(self, name="store/base"):
        pretrained_dict = torch.load(name + ".pt")
        print(bcolors.OKBLUE + "Loaded", name + "." + bcolors.ENDC)
        model_dict = self.model.state_dict()
        model_dict.update(pretrained_dict)
        self.model.load_state_dict(model_dict)


class GanSupervisor():
    def __init__(self, model, discriminator, dataset, loss=nn.BCELoss(reduction='mean'), fake_loss=None):
        self.model = model
        self.discriminator = discriminator
        self.dataset = dataset
        self.loss = loss
        self.fake_loss = fake_loss

    def supervise(self, lr=1e-3, optimizer=torch.optim.Adam, epochs=10, batch_size=32, shuffle=True,
                  num_workers=0, name="store/base", pretrained=False):
        print(bcolors.OKGREEN + "Train with " +
              type(self).__name__ + bcolors.ENDC)
        try:
            if pretrained:
                self.load(name)
        except Exception:
            raise IOError("Could not load pretrained.")
        try:
            train_loader = torch.utils.data.DataLoader(self.dataset, batch_size=batch_size,
                                                       shuffle=shuffle, num_workers=num_workers)
            optimizer_m = optimizer(self.model.parameters(), lr=lr)
            optimizer_d = optimizer(self.discriminator.parameters(), lr=lr)

            for epoch_id in range(epochs):
                loss_d_sum = 0
                loss_m_sum = 0
                tkb = tqdm(total=int(len(train_loader)), bar_format="{l_bar}%s{bar}%s{r_bar}" % (
                    Fore.GREEN, Fore.RESET), desc="Batch Process Epoch " + str(epoch_id))

                for batch_id, data in enumerate(train_loader):
                    inputs, labels = data
                    inputs, labels = inputs.to('cuda'), labels.to('cuda')

                    # Discriminator ++++++++++++++++++++++++++
                    # Disc with real
                    optimizer_d.zero_grad()
                    outputs = self.discriminator(labels)
                    loss_r = self.loss(
                        outputs, torch.ones(outputs.shape).to('cuda'))
                    loss_r.backward()

                    # Disc with fake
                    fakes = self.model(inputs)
                    outputs = self.discriminator(fakes.detach())
                    loss_f = self.loss(
                        outputs, torch.zeros(outputs.shape).to('cuda'))
                    loss_f.backward()

                    loss_d = loss_r + loss_f
                    loss_d_sum += loss_d.item()
                    optimizer_d.step()

                    # Generator +++++++++++++++++++++++++++++
                    optimizer_m.zero_grad()
                    outputs = self.discriminator(fakes)
                    loss_m = self.loss(
                        outputs, torch.ones(outputs.shape).to('cuda'))
                    if self.fake_loss is not None:
                        loss_m += self.fake_loss(fakes, labels)
                    loss_m.backward()
                    optimizer_m.step()

                    loss_m_sum += loss_m.item()
                    tkb.set_postfix(loss_d='{:3f}'.format(
                        loss_d_sum / (batch_id+1)), loss_m='{:3f}'.format(
                        loss_m_sum / (batch_id+1)))
                    tkb.update(1)
        finally:
            self.save(name)
            print()

    def to(self, name):
        self.model = self.model.to(name)
        self.discriminator = self.discriminator.to(name)
        return self

    def get_backbone(self):
        return self.model.backbone

    def save(self, name="store/base"):
        torch.save(self.model.state_dict(), name + ".pt")
        print(bcolors.OKBLUE + "Saved at " + name + "." + bcolors.ENDC)

    def load(self, name="store/base"):
        pretrained_dict = torch.load(name + ".pt")
        print(bcolors.OKBLUE + "Loaded", name + "." + bcolors.ENDC)
        model_dict = self.model.state_dict()
        model_dict.update(pretrained_dict)
        self.model.load_state_dict(model_dict)


class LabelSupervisor(Supervisor):
    def __init__(self, model, dataset, loss=nn.CrossEntropyLoss(reduction='mean')):
        super().__init__(model, dataset, loss)


class RotateNetSupervisor(Supervisor):
    def __init__(self, dataset, rotations=[0.0, 90.0, 180.0,  -90.0], backbone=None, predictor=None, loss=nn.CrossEntropyLoss(reduction='mean')):
        super().__init__(CombinedNet(ReshapeChannels(EfficientFeatures())
                                     if backbone is None else backbone,
                                     Classification(
                                         layers=[4096, 1024, 256, len(rotations)])
                                     if predictor is None else predictor),
                         RotateDataset(dataset, rotations=rotations),
                         loss)


class ExemplarNetSupervisor(Supervisor):
    def __init__(self, dataset, transformations=['rotation', 'crop', 'gray', 'flip', 'erase'], n_classes=8000, n_trans=100, max_elms=10, p=0.5,
                 backbone=None, predictor=None, loss=nn.CrossEntropyLoss(reduction='mean')):
        super().__init__(CombinedNet(ReshapeChannels(EfficientFeatures())
                                     if backbone is None else backbone,
                                     Classification(
                                         layers=[4096, 1024, 1024, n_classes])
                                     if predictor is None else predictor),
                         ExemplarDataset(
                             dataset, transformations=transformations, n_classes=n_classes, n_trans=n_trans, max_elms=max_elms, p=p),
                         loss)


class JigsawNetSupervisor(Supervisor):
    # Not the CFN of the paper for easier implementation with common backbones and, most importantly, easier reuse
    def __init__(self, dataset, jigsaw_path="utils/permutations_hamming_max_1000.npy", n_perms_per_image=69, crop_size=64,
                 backbone=None, predictor=None, loss=nn.CrossEntropyLoss(reduction='mean')):
        super().__init__(CombinedNet(ReshapeChannels(EfficientFeatures())
                                     if backbone is None else backbone,
                                     Classification(
                                         layers=[4096, 1024, 1024, 1000])
                                     if predictor is None else predictor),
                         JigsawDataset(
                             dataset, jigsaw_path="utils/permutations_hamming_max_1000.npy", n_perms_per_image=69, crop_size=64),
                         loss)


class DenoiseNetSupervisor(Supervisor):
    # Not the CFN of the paper for easier implementation with common backbones and, most importantly, easier reuse
    def __init__(self, dataset, p=0.7,
                 backbone=None, predictor=None, loss=nn.MSELoss(reduction='mean')):
        super().__init__(CombinedNet(EfficientFeatures()
                                     if backbone is None else backbone,
                                     Upsampling(
                                         layers=[1280, 512, 256, 128, 64, 3])
                                     if predictor is None else predictor),
                         DenoiseDataset(dataset, p=p),
                         loss)


class SplitBrainNetSupervisor(Supervisor):
    # Not the CFN of the paper for easier implementation with common backbones and, most importantly, easier reuse
    def __init__(self, dataset, l_step=2, l_offset=0, ab_step=26, a_offset=128, b_offset=128,
                 backbone=None, predictor=None, loss=GroupedCrossEntropyLoss()):
        super().__init__(CombinedNet(GroupedEfficientFeatures()
                                     if backbone is None else backbone,
                                     GroupedUpsampling(
                                         layers=[1280, 512, 256, 128, 64])
                                     if predictor is None else predictor),
                         SplitBrainDataset(dataset, l_step=l_step, l_offset=l_offset - 1,
                                           ab_step=ab_step, a_offset=a_offset, b_offset=b_offset),
                         loss)


class ContextNetSupervisor(GanSupervisor):
    # Not the CFN of the paper for easier implementation with common backbones and, most importantly, easier reuse
    def __init__(self, dataset,  p=0.3, n_blocks=10, scale_range=(0.05, 0.1),
                 backbone=None, predictor=None, discriminator=None, loss=nn.MSELoss(reduction='mean'), fake_loss=nn.MSELoss(reduction='mean')):
        super().__init__(CombinedNet(ChannelwiseFC(EfficientFeatures())
                                     if backbone is None else backbone,
                                     Upsampling(
                                         layers=[1280, 512, 256, 128, 64, 3])
                                     if predictor is None else predictor),
                         CombinedNet(ReshapeChannels(EfficientFeatures()), Classification(
                             layers=[4096,  1024, 256, 1])) if discriminator is None else discriminator,
                         ContextDataset(
                             dataset, p=p, n_blocks=n_blocks, scale_range=scale_range),
                         loss,
                         fake_loss)


class BiGanSupervisor(GanSupervisor):
    # Not the CFN of the paper for easier implementation with common backbones and, most importantly, easier reuse
    def __init__(self, dataset, shape=(32, 8, 8), rand_gen=np.random.rand,
                 backbone=None, predictor=None, discriminator=None, loss=nn.MSELoss(reduction='mean'), fake_loss=nn.MSELoss(reduction='mean')):
        super().__init__(CombinedNet(ReshapeChannels(EfficientFeatures(), out_channels=32)
                                     if backbone is None else backbone,
                                     Upsampling(
                                         layers=[32, 128, 256, 128, 64, 3])
                                     if predictor is None else predictor),
                         CombinedNet(ReshapeChannels(EfficientFeatures(), out_channels=32), Classification(
                             layers=[4096, 1024, 256, 1])) if discriminator is None else discriminator,
                         BiDataset(
                             dataset, shape=shape, rand_gen=rand_gen),
                         loss,
                         fake_loss)

    def supervise(self, lr=1e-3, optimizer=torch.optim.Adam, epochs=10, batch_size=32, shuffle=True,
                  num_workers=0, name="store/base", pretrained=False):
        print(bcolors.OKGREEN + "Train with " +
              type(self).__name__ + bcolors.ENDC)
        try:
            if pretrained:
                self.load(name)
        except Exception:
            raise IOError("Could not load pretrained.")
        try:
            train_loader = torch.utils.data.DataLoader(self.dataset, batch_size=batch_size,
                                                       shuffle=shuffle, num_workers=num_workers)
            optimizer_m = optimizer(self.model.parameters(), lr=lr)
            optimizer_d = optimizer(self.discriminator.parameters(), lr=lr)

            for epoch_id in range(epochs):
                loss_d_sum = 0
                loss_m_sum = 0
                tkb = tqdm(total=int(len(train_loader)), bar_format="{l_bar}%s{bar}%s{r_bar}" % (
                    Fore.GREEN, Fore.RESET), desc="Batch Process Epoch " + str(epoch_id))

                for batch_id, data in enumerate(train_loader):
                    inputs, labels = data
                    inputs, labels = inputs.to('cuda'), labels.to('cuda')

                    # Discriminator ++++++++++++++++++++++++++
                    # Disc with real
                    optimizer_d.zero_grad()
                    joined_in = torch.cat((self.model.backbone(
                        labels), self.discriminator.backbone(labels)), dim=1)
                    outputs = self.discriminator.predictor(joined_in)
                    loss_r = self.loss(
                        outputs, torch.ones(outputs.shape).to('cuda'))
                    loss_r.backward()

                    # Disc with fake
                    fakes = self.model.predictor(inputs)
                    joined_in = torch.cat(
                        (inputs.reshape(inputs.shape[0], -1), self.discriminator.backbone(fakes.detach())), dim=1)
                    outputs = self.discriminator.predictor(joined_in)
                    loss_f = self.loss(
                        outputs, torch.zeros(outputs.shape).to('cuda'))
                    loss_f.backward()

                    loss_d = loss_r + loss_f
                    loss_d_sum += loss_d.item()
                    optimizer_d.step()

                    # Generator +++++++++++++++++++++++++++++
                    optimizer_m.zero_grad()
                    joined_in = torch.cat(
                        (inputs.reshape(inputs.shape[0], -1), self.discriminator.backbone(fakes.detach())), dim=1)
                    outputs = self.discriminator.predictor(joined_in)
                    loss_m = self.loss(
                        outputs, torch.ones(outputs.shape).to('cuda'))
                    if self.fake_loss is not None:
                        loss_m += self.fake_loss(fakes, labels)
                    loss_m.backward()
                    optimizer_m.step()

                    loss_m_sum += loss_m.item()
                    tkb.set_postfix(loss_d='{:3f}'.format(
                        loss_d_sum / (batch_id+1)), loss_m='{:3f}'.format(
                        loss_m_sum / (batch_id+1)))
                    tkb.update(1)
        finally:
            self.save(name)
            print()


class ContrastivePredictiveCodingSupervisor(Supervisor):
    # Not the CFN of the paper for easier implementation with common backbones and, most importantly, easier reuse
    def __init__(self, dataset, half_crop_size=(int(25), int(25)),
                 backbone=None, predictor=None, loss=InfoNCE(k=3, ignore=2).to('cuda')):
        super().__init__(CombinedNet(Batch2Image(nn.Sequential(EfficientFeatures(), nn.AvgPool2d(2)))
                                     if backbone is None else backbone,
                                     ReshapeChannels(MaskedCNN(
                                         layers=[1280, 512, 256, 128, 128], mask=torch.from_numpy(np.array([[1, 1, 1], [1, 1, 1], [0, 0, 0]]))),
                                         in_channels=128, out_channels=64 * loss.k, kernel_size=1, padding=0, activation=nn.Identity, flat=False)
                                     if predictor is None else predictor),
                         ContrastivePreditiveCodingDataset(
                             dataset, half_crop_size=half_crop_size),
                         loss)

    def supervise(self, lr=1e-3, optimizer=torch.optim.Adam, epochs=10, batch_size=32, shuffle=True,
                  num_workers=0, name="store/base", pretrained=False, collate_fn=None):
        print(bcolors.OKGREEN + "Train with " +
              type(self).__name__ + bcolors.ENDC)
        try:
            if pretrained:
                self.load(name)
        except Exception:
            raise IOError("Could not load pretrained.")
        try:
            train_loader = torch.utils.data.DataLoader(self.dataset, batch_size=batch_size,
                                                       shuffle=shuffle, num_workers=num_workers, collate_fn=collate_fn)
            optimizer = optimizer(self.model.parameters(), lr=lr)
            for epoch_id in range(epochs):
                loss_sum = 0
                tkb = tqdm(total=int(len(train_loader)), bar_format="{l_bar}%s{bar}%s{r_bar}" % (
                    Fore.GREEN, Fore.RESET), desc="Batch Process Epoch " + str(epoch_id))

                for batch_id, data in enumerate(train_loader):
                    inputs, _ = data
                    optimizer.zero_grad()
                    encodings = self.model.backbone(inputs.to('cuda'))
                    predictions = self.model.predictor(encodings)
                    loss = self.loss(predictions, encodings)
                    loss.backward()
                    optimizer.step()
                    loss_sum += loss.item()
                    tkb.set_postfix(loss='{:3f}'.format(
                        loss_sum / (batch_id+1)))
                    tkb.update(1)
        finally:
            self.save(name)
            print()
