import torch
from torch import nn
from torch.nn import functional as F
from .data import visualize, RotateDataset, ExemplarDataset, JigsawDataset, DenoiseDataset, \
    ContextDataset, BiDataset, SplitBrainDataset, ContrastivePreditiveCodingDataset, AugmentationDataset, \
    AugmentationIndexedDataset, AugmentationLabIndexedDataset
from .models import ReshapeChannels, Classification, Batch2Image, GroupedLoss, \
    CPCLoss, MaskedCNN, EfficientFeatures, CombinedNet, Upsampling, ChannelwiseFC, GroupedEfficientFeatures, \
    GroupedUpsampling, SequentialUpTo, View
from tqdm import tqdm
from colorama import Fore
from .utils import bcolors
import numpy as np
import copy
from .data import batched_collate, visualize
from .data import ContrastivePredictiveCodingAugmentations, MomentumContrastAugmentations, BYOLAugmentations, PIRLAugmentations
from .memory import BatchedQueue, BatchedMemory


class Supervisor():

    def __init__(self, model, dataset=None, loss=nn.CrossEntropyLoss(reduction='mean'), collate_fn=None):
        """Constitutes a self-supervision algorithm. All implemented algorithms are childs. Handles training, storing, and
        loading of the trained model/backbone.

        Args:
            model (torch.nn.Module): The module to self supervise.
            dataset (torch.utils.data.Dataset): The dataset to train on.
            loss (torch.nn.Module, optional): The critierion to train on. Defaults to nn.CrossEntropyLoss(reduction='mean').
            collate_fn (function, optional): The collate function. Defaults to None.
        """
        if not isinstance(model, CombinedNet):
            raise("You must pass a CombinedNet to model.")
        self.model = nn.DataParallel(model)
        self.dataset = dataset
        self.loss = loss
        self.collate_fn = collate_fn
        self.memory=None

    def supervise(self, lr=1e-3, optimizer=torch.optim.Adam, epochs=10, batch_size=32, shuffle=True,
                  num_workers=0, name="store/base", pretrained=False, lr_scheduler=lambda optimizer: torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=1.0)):
        """Starts the training procedure of a self-supervision algorithm.

        Args:
            lr (float, optional): Optimizer learning rate. Defaults to 1e-3.
            lr_scheduler (torch.optim._LRScheduler, optional): Optional learning rate scheduler. Defaults to lambdaoptimizer:torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=1.0).
            optimizer (torch.optim.Optimizer, optional): Optimizer to use. Defaults to torch.optim.Adam.
            epochs (int, optional): Number of epochs to train. Defaults to 10.
            batch_size (int, optional): Size of bachtes to process. Defaults to 32.
            shuffle (bool, optional): Wether to shuffle the dataset. Defaults to True.
            num_workers (int, optional): Number of workers to use. Defaults to 0.
            name (str, optional): Path to store and load models. Defaults to "store/base".
            pretrained (bool, optional): Wether to load pretrained model. Defaults to False.
        """
        if not isinstance(self.dataset, torch.utils.data.Dataset):
            raise("No dataset has been specified.")
        print(bcolors.OKGREEN + "Train with " +
              type(self).__name__ + bcolors.ENDC)
        self._load_pretrained(name, pretrained)
        try:
            train_loader, optimizer, lr_scheduler = self._init_data_optimizer(
                optimizer=optimizer, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=self.collate_fn, lr=lr, lr_scheduler=lr_scheduler)
            self._epochs(epochs=epochs, train_loader=train_loader,
                         optimizer=optimizer, lr_scheduler=lr_scheduler)
        finally:
            self.save(name)
            print()

    def _load_pretrained(self, name, pretrained):
        """Private method to load a pretrained model

        Args:
            name (str): Path to model.
            pretrained (bool): Wether to load pretrained model.

        Raises:
            IOError: [description]
        """
        try:
            if pretrained:
                self.load(name)
        except Exception:
            raise IOError("Could not load pretrained.")

    def _init_data_optimizer(self, optimizer, batch_size, shuffle, num_workers, collate_fn, lr, lr_scheduler):
        """Creates all objects that are neccessary for the self-supervision training and are dependend on self.supervise(...).

        Args:
            optimizer (torch.optim.Optimizer, optional): Optimizer to use.
            batch_size (int, optional): Size of bachtes to process.
            shuffle (bool, optional): Wether to shuffle the dataset.
            num_workers (int, optional): Number of workers to use.
            collate_fn (function, optional): The collate function.
            lr (float, optional): Optimizer learning rate.
            lr_scheduler (torch.optim._LRScheduler, optional): Optional learning rate scheduler.

        Returns:
            Tuple: All created objects
        """
        train_loader = torch.utils.data.DataLoader(self.dataset, batch_size=batch_size,
                                                   shuffle=shuffle, num_workers=num_workers, collate_fn=collate_fn)
        optimizer = optimizer(self.model.parameters(), lr=lr)
        lr_scheduler = lr_scheduler(optimizer)

        return train_loader, optimizer, lr_scheduler

    def _epochs(self, epochs, train_loader, optimizer, lr_scheduler):
        """Implements the training loop (epochs, batches, visualization) excluding the actual training step.

        Args:
            epochs (int, optional): Number of epochs to train.
            train_loader (torch.utils.data.DataLoader): Iterator over the dataset.
            lr_scheduler (torch.optim._LRScheduler, optional): Optional learning rate scheduler.
            optimizer (torch.optim.Optimizer, optional): Optimizer to use.
        """
        tkb = tqdm(total=int(len(train_loader)), bar_format="{l_bar}%s{bar}%s{r_bar}" % (
            Fore.GREEN, Fore.RESET))
        for epoch_id in range(epochs):
            loss_sum = 0
            tkb.set_description(desc="Batch Process Epoch " + str(epoch_id))
            for batch_id, data in enumerate(train_loader):
                if data[0].shape[0] != train_loader.batch_size and batch_id == (len(train_loader) - 1):
                    continue
                optimizer.zero_grad()
                loss = self._forward(data)
                loss_sum += loss.item()
                tkb.set_postfix(loss='{:3f}'.format(
                    loss_sum / (batch_id+1)))
                tkb.update(1)

                self._update(loss=loss, optimizer=optimizer,
                             lr_scheduler=lr_scheduler)
            tkb.reset()

    def _forward(self, data):
        """Forward part of training step. Conducts all forward calculations.

        Args:
            data (Tuple(torch.FloatTensor,torch.FloatTensor)): Batch of instances with corresponding labels.

        Returns:
            torch.FloatTensor: Loss of batch.
        """
        inputs, labels = data
        outputs = self.model(inputs.to('cuda'))
        loss = self.loss(outputs, labels.to('cuda'))
        return loss

    def _update(self, loss, optimizer, lr_scheduler):
        """Backward part of training step. Calculates gradients and conducts optimization step.
        Also handles other updates like lr scheduler.

        Args:
            loss (torch.nn.Module, optional): The critierion to train on.
            lr_scheduler (torch.optim._LRScheduler, optional): Optional learning rate scheduler.
            optimizer (torch.optim.Optimizer, optional): Optimizer to use.
        """
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

    def to(self, name):
        """Wraps device handling.

        Args:
            name (str): Name of device, see pytorch.

        Returns:
            Supervisor: Returns itself.
        """
        self.model = self.model.to(name)
        return self

    def get_backbone(self):
        """Extracts the backbone network that creates features

        Returns:
            torch.nn.Module: The backbone network.
        """
        try:
            return self.model.module.backbone
        except:
            return self.model.backbone

    def get_predictor(self):
        """Extracts the predictor network

        Returns:
            torch.nn.Module: The backbone network.
        """
        try:
            return self.model.module.predictor
        except:
            return self.model.predictor

    def save(self, name="store/base"):
        """Saves model parameters to disk.

        Args:
            name (str, optional): Path to storage. Defaults to "store/base".
        """
        torch.save(self.model.module.state_dict(), name + ".pt")
        if self.memory is not None:
            self.memory.save(name + "_memory" + ".pt")
        print(bcolors.OKBLUE + "Saved at " + name + "." + bcolors.ENDC)
        return self

    def load(self, name="store/base"):
        """Loads model parameters from disk.

        Args:
            name (str, optional): Path to storage. Defaults to "store/base".
        """
        pretrained_dict = torch.load(name + ".pt")
        print(bcolors.OKBLUE + "Loaded", name + "." + bcolors.ENDC)
        model_dict = self.model.module.state_dict()
        model_dict.update(pretrained_dict)
        self.model.module.load_state_dict(model_dict)
        if self.memory is not None:
            self.memory.load(name + "_memory" + ".pt")
        return self


class GanSupervisor():
    def __init__(self, model, discriminator, dataset, loss=nn.BCELoss(reduction='mean'), fake_loss=None):
        self.model = nn.DataParallel(model)
        self.discriminator = nn.DataParallel(discriminator)
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
        try:
            return self.model.module.backbone
        except:
            return self.model.backbone


    def get_predictor(self):
        try:
            return self.model.module.predictor
        except:
            return self.model.predictor

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
        """Same as supervisor but does not backbrop through backbone.

        Args:
            model (torch.nn.Module): The module to self supervise.
            dataset (torch.utils.data.Dataset): The dataset to train on.
            loss (torch.nn.Module, optional): The critierion to train on. Defaults to nn.CrossEntropyLoss(reduction='mean').
        """
        super().__init__(model, dataset, loss)
        for _, current in enumerate(self.model.module.backbone.parameters()):
            current.requires_grad = False


class RotateNetSupervisor(Supervisor):
    def __init__(self, dataset=None, rotations=[0.0, 90.0, 180.0,  -90.0], backbone=None, predictor=None, loss=nn.CrossEntropyLoss(reduction='mean')):
        """RotateNet like supervisor https://arxiv.org/abs/1803.07728.

        Args:
            dataset (torch.utils.data.Dataset): The dataset to train on.
            rotations (list, optional): Rotations to predict. Defaults to [0.0, 90.0, 180.0,  -90.0].
            backbone (torch.nn.Module, optional): Backbone network. Defaults to None, resulting in an EfficientNet backbone.
            predictor (torch.nn.Module, optional): Prediction network. Defaults to None, resulting in a MLP that fits to the number of rotations to predict.
            loss (torch.nn.Module, optional): The critierion to train on. Defaults to nn.CrossEntropyLoss(reduction='mean').
        """
        super().__init__(CombinedNet(ReshapeChannels(EfficientFeatures())
                                     if backbone is None else backbone,
                                     Classification(
                                         layers=[3136, 1024, 256, len(rotations)])
                                     if predictor is None else predictor),
                         RotateDataset(dataset, rotations=rotations),
                         loss,
                         collate_fn=batched_collate)


class ExemplarNetSupervisor(Supervisor):
    def __init__(self, dataset=None, n_classes=8000, n_trans=100, max_elms=10, p=0.5,
                 backbone=None, predictor=None, loss=nn.CrossEntropyLoss(reduction='mean')):
        """ExemplarNet like supervisor https://arxiv.org/abs/1406.6909.

        Args:
            dataset (torch.utils.data.Dataset): The dataset to train on.
            n_classes (int, optional): Number of classes, i.e. the subset size of the dataset. Defaults to 8000.
            n_trans (int, optional): Number of combined transformations. Defaults to 100.
            max_elms (int, optional): Number of elementar transformations per combined transformation. Defaults to 10.
            p (float, optional): Prob. of an elmentar transformation to be part of a combined transformation. Defaults to 0.5.
            backbone (torch.nn.Module, optional): Backbone network. Defaults to None, resulting in an EfficientNet backbone.
            predictor (torch.nn.Module, optional): Prediction network. Defaults to None, resulting in a MLP that fits to the number of classes to predict.
            loss (torch.nn.Module, optional): The critierion to train on. Defaults to nn.CrossEntropyLoss(reduction='mean').
        """
        super().__init__(CombinedNet(ReshapeChannels(EfficientFeatures())
                                     if backbone is None else backbone,
                                     Classification(
                                         layers=[3136, 1024, 1024, n_classes])
                                     if predictor is None else predictor),
                         ExemplarDataset(
                             dataset, n_classes=n_classes, n_trans=n_trans, max_elms=max_elms, p=p),
                         loss)


class JigsawNetSupervisor(Supervisor):
    # Not the CFN of the paper for easier implementation with common backbones and, most importantly, easier reuse
    def __init__(self, dataset=None, jigsaw_path="super_selfish/utils/four_perms.npy", n_perms_per_image=24, total_perms=24, crops=2, crop_size=112,
                 backbone=None, predictor=None, loss=nn.CrossEntropyLoss(reduction='mean')):
        """Jigsaw puzzle like supervisor https://arxiv.org/abs/1603.09246.
            We use layer norm.

        Args:
            dataset (torch.utils.data.Dataset): The dataset to train on.
            jigsaw_path (str, optional): The path to the used permutations. Defaults to "super_selfish/utils/four_perms.npy".
            n_perms_per_image (int, optional): Number of permutations per image. Defaults to 24.
            total_perms (int, optional): Number of perms in total. Defaults to 24.
            crops (int, optional): Number of patches is crops x crops. Defaults to 2.
            crop_size (int, optional): Crop size, implicitly determines the distance between crops. Defaults to 112.
            backbone (torch.nn.Module, optional): Backbone network. Defaults to None, resulting in an EfficientNet backbone.
            predictor (torch.nn.Module, optional): Prediction network. Defaults to None, resulting in a MLP that fits to the number of permutations used.
            loss (torch.nn.Module, optional): The critierion to train on. Defaults to nn.CrossEntropyLoss(reduction='mean').
        """
        super().__init__(CombinedNet(ReshapeChannels(EfficientFeatures(norm_type='layer'))
                                     if backbone is None else backbone,
                                     Classification(
                                         layers=[3136, 1024, 1024, total_perms])
                                     if predictor is None else predictor),
                         JigsawDataset(
                             dataset, jigsaw_path=jigsaw_path, n_perms_per_image=n_perms_per_image, total_perms=total_perms, crops=crops, crop_size=crop_size),
                         loss)


class DenoiseNetSupervisor(Supervisor):
    def __init__(self, dataset=None, p=0.7,
                 backbone=None, predictor=None, loss=nn.MSELoss(reduction='mean')):
        """Denoising autoencoder https://www.cs.toronto.edu/~larocheh/publications/icml-2008-denoising-autoencoders.pdf.

        Args:
            dataset (torch.utils.data.Dataset): The dataset to train on.
            p (float, optional): Noise level. Defaults to 0.7.
            backbone (torch.nn.Module, optional): Backbone network. Defaults to None, resulting in an EfficientNet backbone.
            predictor (torch.nn.Module, optional): Prediction network. Defaults to None, resulting in a standard upsampling ConvNet.
            loss ([type], optional): The critierion to train on. Defaults to nn.MSELoss(reduction='mean').
        """
        super().__init__(CombinedNet(ReshapeChannels(EfficientFeatures(), flat=False)
                                     if backbone is None else backbone,
                                     Upsampling(
                                         layers=[64, 256, 256, 128, 64, 3])
                                     if predictor is None else predictor),
                         DenoiseDataset(dataset, p=p),
                         loss)


class SplitBrainNetSupervisor(Supervisor):
    def __init__(self, dataset=None, l_step=2, l_offset=0, ab_step=26, a_offset=128, b_offset=128,
                 backbone=None, predictor=None, loss=GroupedLoss()):
        """Splitbrain autoencoder https://arxiv.org/pdf/1611.09842.pdf.

        Args:
            dataset (torch.utils.data.Dataset): The dataset to train on.
            l_step (int, optional): l channel bin size. Defaults to 2.
            l_offset (int, optional): l channel offset. Defaults to 0.
            ab_step (int, optional): ab channel bin size Defaults to 26.
            a_offset (int, optional): a channel offset. Defaults to 128.
            b_offset (int, optional): b channel offset. Defaults to 128.
            backbone (torch.nn.Module, optional): Backbone network. Defaults to None, resulting in an EfficientNet backbone.
            predictor (torch.nn.Module, optional): Prediction network. Defaults to None, resulting in a standard upsampling ConvNet.
            loss ([type], optional): The critierion to train on. Defaults to GroupedLoss().
        """
        super().__init__(CombinedNet(GroupedEfficientFeatures()
                                     if backbone is None else backbone,
                                     GroupedUpsampling(
                                         layers=[32, 256, 256, 128, 64])
                                     if predictor is None else predictor),
                         SplitBrainDataset(dataset, l_step=l_step, l_offset=l_offset - 1,
                                           ab_step=ab_step, a_offset=a_offset, b_offset=b_offset),
                         loss)


class ContextNetSupervisor(GanSupervisor):
    def __init__(self, dataset=None,  p=0.3, n_blocks=10, scale_range=(0.05, 0.1),
                 backbone=None, predictor=None, discriminator=None, loss=nn.MSELoss(reduction='mean'), fake_loss=nn.MSELoss(reduction='mean')):
        """Context autoencoder https://arxiv.org/pdf/1604.07379.pdf .

        Args:
            dataset (torch.utils.data.Dataset): The dataset to train on.
            p (float, optional): Prob. with which a block is erased. Defaults to 0.3.
            n_blocks (int, optional): Number of blocks that may be erased in a given image. Defaults to 10.
            scale_range (tuple, optional): Block scale range from which a scale is sampled per block. Defaults to (0.05, 0.1).
            backbone (torch.nn.Module, optional): Backbone network. Defaults to None, resulting in an EfficientNet backbone.
            predictor (torch.nn.Module, optional): Prediction network. Defaults to None, resulting in a standard upsampling ConvNet.
            discriminator (torch.nn.Module, optional): Discriminator network. Defaults to None, resulting in an EfficientNet.
            loss ([type], optional): The critierion to train on.. Defaults to nn.MSELoss(reduction='mean').
            fake_loss ([type], optional): The advesarial criterion. Defaults to nn.MSELoss(reduction='mean').
        """
        super().__init__(CombinedNet(ReshapeChannels(EfficientFeatures(), flat=False)
                                     if backbone is None else backbone,
                                     nn.Sequential(ChannelwiseFC(),
                                     Upsampling(
                                         layers=[64, 256, 256, 128, 64, 3]))
                                     if predictor is None else predictor),
                         CombinedNet(ReshapeChannels(EfficientFeatures()), Classification(
                             layers=[3136,  1024, 256, 1])) if discriminator is None else discriminator,
                         ContextDataset(
                             dataset, p=p, n_blocks=n_blocks, scale_range=scale_range),
                         loss,
                         fake_loss)


class BiGanSupervisor(GanSupervisor):
    def __init__(self, dataset=None, shape=(32, 7, 7), rand_gen=np.random.rand,
                 backbone=None, predictor=None, discriminator=None, loss=nn.MSELoss(reduction='mean'), fake_loss=nn.MSELoss(reduction='mean')):
        """BiGan Supervisor https://arxiv.org/pdf/1605.09782.pdf.
        Args:
            dataset (torch.utils.data.Dataset): The dataset to train on.
            shape (tuple, optional): Latent vector shape. Defaults to (32, 7, 7).
            rand_gen (np.random, optional): Random noise distribution. Defaults to np.random.rand.
            backbone (torch.nn.Module, optional): Backbone network. Defaults to None, resulting in an EfficientNet backbone.
            predictor (torch.nn.Module, optional): Prediction network. Defaults to None, resulting in a standard upsampling ConvNet.
            discriminator (torch.nn.Module, optional): Discriminator network. Defaults to None, resulting in an EfficientNet.
            loss ([type], optional): The critierion to train on.. Defaults to nn.MSELoss(reduction='mean').
            fake_loss ([type], optional): The advesarial criterion. Defaults to nn.MSELoss(reduction='mean').
        """
        super().__init__(CombinedNet(ReshapeChannels(EfficientFeatures(), out_channels=32)
                                     if backbone is None else backbone,
                                     Upsampling(
                                         layers=[32, 128, 256, 128, 64, 3])
                                     if predictor is None else predictor),
                         CombinedNet(ReshapeChannels(EfficientFeatures(), out_channels=32), Classification(
                             layers=[3136, 1024, 256, 1])) if discriminator is None else discriminator,
                         BiDataset(
                             dataset, shape=shape, rand_gen=rand_gen),
                         loss,
                         fake_loss)

        self.model = self.model.module
        self.discriminator = self.discriminator.module

    def supervise(self, lr=1e-3, optimizer=torch.optim.Adam, epochs=10, batch_size=32, shuffle=True,
                  num_workers=0, name="store/base", pretrained=False, collate_fn=None, lr_scheduler=lambda optimizer: torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=1.0)):
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
    def __init__(self, dataset=None, embedding_size=128, half_crop_size=(int(28), int(28)), sides=['top', 'bottom', 'left', 'right'], bottleneck_resolution = (7,7),
                 backbone=None, predictor=None, loss=CPCLoss(k=3, ignore=2).to('cuda')):
        """Contrastive Predictive Coding v2 for future prediction https://arxiv.org/pdf/1905.09272.pdf.

        Args:
            dataset (torch.utils.data.Dataset): The dataset to train on.
            embedding_size (int, optional): Size of predicted embeddings. Defaults to 128.
            half_crop_size (tuple, optional): Half size of crops to predict. Defaults to (int(28), int(28)).
            sides (list, optional): From which sides to generate context. Defaults to ['top', 'bottom', 'left', 'right'].
            bottleneck_resolution ((int, int), optional): Determines the 2D shape of the bottleneck. Defaults to (7,7).
            backbone (torch.nn.Module, optional): Backbone network. Defaults to None, resulting in an EfficientNet backbone with LayerNorm.
            predictor (torch.nn.Module, optional): Prediction network. Defaults to None, resulting in a MaskedCNN and a linear predictor for each side.
            loss (torch.nn.Module, optional): The critierion to train on. Defaults to CPCLoss(k=3, ignore=2), a specific CPC loss for efficient calculation.
        """
        super().__init__(CombinedNet(ReshapeChannels(EfficientFeatures(norm_type='layer'), flat=False)
                                     if backbone is None else backbone,
                                     nn.ModuleDict({side: ReshapeChannels(MaskedCNN(
                                         layers=[64, 256, 256, 128, embedding_size], mask=torch.from_numpy(np.array([[1, 1, 1], [1, 1, 1], [0, 0, 0]])),  side=side),
                                         in_channels=128, out_channels=64 * loss.k, kernel_size=1, padding=0, activation=nn.Identity, flat=False) for side in sides})
                                     if predictor is None else predictor),
                         ContrastivePreditiveCodingDataset(
                             dataset, half_crop_size=half_crop_size),
                         loss,
                         batched_collate)
        self.batch2image = nn.DataParallel(Batch2Image(bottleneck_resolution))
        self.backbone_para = nn.DataParallel(self.get_backbone())
        self.predictor_para = nn.ModuleDict({side: nn.DataParallel(self.get_predictor()[side]) for side in sides})
        self.sides = sides

    def _forward(self, data):
        inputs, _ = data
        encodings = self.backbone_para(inputs.to('cuda'))
        encodings = self.batch2image(encodings)
        loss = 0
        for side in self.sides:
            predictions = self.predictor_para[side](encodings)
            if side == 'top':
                encodings_a = encodings
            elif side == 'bottom':
                encodings_a = torch.flip(encodings, dims=[2]).contiguous()
            elif side == 'right':
                encodings_a = torch.transpose(encodings, 2, 3).contiguous()
            elif side == 'left':
                encodings_a = torch.flip(
                    torch.transpose(encodings, 2, 3).contiguous(), dims=[3]).contiguous()

            loss += self.loss(predictions, encodings_a)
        return loss

    def _update(self, loss, optimizer, lr_scheduler):
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), 0.01)
        optimizer.step()
        lr_scheduler.step()


class MomentumContrastSupervisor(Supervisor):
    def __init__(self, dataset=None, data_augmentation=lambda dataset : AugmentationDataset(dataset, transformations=MomentumContrastAugmentations), embedding_size=128, K=8, m=0.999,  t=0.2,
                 backbone=None, predictor=None, loss=nn.CrossEntropyLoss(reduction='mean')):
        """Momentum contrast v2 https://arxiv.org/pdf/2003.04297.pdf.

        Args:
            dataset (torch.utils.data.Dataset): The dataset to train on.
            data_augmentation (lambda dataset: dataset, optional): Lambda function that returns a dataset with augmentations. Defaults to lambda dataset : AugmentationDataset(dataset, transformations=MomentumContrastAugmentations).
            embedding_size (int, optional): Size of predicted embeddings. Defaults to 128.
            K (int, optional): Size of queue in batches. Defaults to 8.
            m (float, optional): Momentum encoder weighting parameter. Defaults to 0.999.
            t (float, optional): Temperature. Defaults to 0.2.
            backbone (torch.nn.Module, optional): Backbone network. Defaults to None, resulting in an EfficientNet backbone.
            predictor (torch.nn.Module, optional): Prediction network. Defaults to None, resulting in a MLP that fits to the embeddings size.
            loss ([type], optional): The critierion to train on. Defaults to nn.CrossEntropyLoss(reduction='mean').
        """
        super().__init__(CombinedNet(ReshapeChannels(EfficientFeatures(norm_type='layer'))
                                     if backbone is None else backbone,
                                     Classification(
                                         layers=[3136, 1024, 1024, embedding_size])
                                     if predictor is None else predictor),
                         data_augmentation(dataset),
                         loss)
        self.embedding_size = embedding_size
        self.K = K
        self.m = m
        self.t = t

    def _epochs(self, epochs, train_loader, optimizer, lr_scheduler):
        tkb = tqdm(total=int(len(train_loader)), bar_format="{l_bar}%s{bar}%s{r_bar}" % (
            Fore.GREEN, Fore.RESET), mininterval=0)

        batch_size = train_loader.batch_size
        self.model_k = copy.deepcopy(self.model)
        # Init queue
        queue = BatchedQueue(K=self.K, batch_size=batch_size,
                             embedding_size=self.embedding_size)
        queue.init_w_loader_and_model(
            train_loader=train_loader, model=self.model_k)
        queue.reset_pointer()

        for epoch_id in range(epochs):
            loss_sum = 0
            tkb.set_description(desc="Batch Process Epoch " + str(epoch_id))
            for batch_id, data in enumerate(train_loader):
                if data[0].shape[0] != train_loader.batch_size:
                    continue
                optimizer.zero_grad()

                loss = self._forward(data, queue)

                loss_sum += loss.item()
                tkb.set_postfix(loss='{:3f}'.format(
                    loss_sum / (batch_id+1)))
                tkb.update(1)

                self._update(loss=loss, optimizer=optimizer,
                             lr_scheduler=lr_scheduler)
                # There should be a better way...
                with torch.no_grad():
                    for param_q, param_k in zip(self.model.parameters(), self.model_k.parameters()):
                        param_k.data = param_k.data * self.m + \
                            param_q.data * (1. - self.m)
            tkb.reset()

    def _forward(self, data, queue):
        imgs1, imgs2 = data
        batch_size = imgs1.shape[0]

        q = self.model(imgs1.to('cuda'))
        with torch.no_grad():
            k = self.model_k(imgs2.to('cuda'))
            k = F.normalize(k)
        q = F.normalize(q)

        l_pos = torch.bmm(q.view(q.shape[0], 1, q.shape[1]), k.view(
            k.shape[0], k.shape[1], 1)).squeeze(2) / self.t

        l_neg = torch.mm(
            q, F.normalize(queue.data()).permute(1, 0)) / self.t
        logits = torch.cat([l_pos, l_neg], dim=1)

        labels = torch.zeros(batch_size, device='cuda').long()
        loss = self.loss(logits, labels)

        with torch.no_grad():
            queue.enqueue(k)

        return loss


class BYOLSupervisor(Supervisor):
    def __init__(self, dataset=None, data_augmentation=lambda dataset : AugmentationDataset(dataset, transformations=BYOLAugmentations), embedding_size=128, m=0.999,
                 backbone=None, predictor=None, loss=nn.CrossEntropyLoss(reduction='mean')):
        """Bootstrap your own latent https://arxiv.org/pdf/2006.07733.pdf.

        Args:
            dataset (torch.utils.data.Dataset): The dataset to train on.
            data_augmentation (lambda dataset: dataset, optional): Lambda function that returns a dataset with augmentations. Defaults to lambda dataset : AugmentationDataset(dataset, transformations=BYOLAugmentations).
            embedding_size (int, optional): Size of predicted embeddings. Defaults to 128.
            m (float, optional): Momentum encoder weighting parameter. Defaults to 0.999.
            backbone (torch.nn.Module, optional): Backbone network. Defaults to None, resulting in an EfficientNet backbone.
            predictor (torch.nn.Module, optional): Prediction network. Defaults to None, resulting in a MLP that fits to the embeddings size.
            loss ([type], optional): The critierion to train on. Defaults to nn.CrossEntropyLoss(reduction='mean').
        """
        super().__init__(CombinedNet(ReshapeChannels(EfficientFeatures())
                                     if backbone is None else backbone,
                                     SequentialUpTo(Classification(layers=[3136, 1024, 1024, embedding_size]), Classification(
                                         layers=[embedding_size, embedding_size * 4, embedding_size * 2, embedding_size]))
                                     if predictor is None else predictor),
                         data_augmentation(dataset),
                         loss)
        self.embedding_size = embedding_size
        self.m = m

    def _epochs(self, epochs, train_loader, optimizer, lr_scheduler):
        tkb = tqdm(total=int(len(train_loader)), bar_format="{l_bar}%s{bar}%s{r_bar}" % (
            Fore.GREEN, Fore.RESET), mininterval=0)
            
        self.model_k = copy.deepcopy(self.model)
        self.model_k.module.predictor.up_to = 0

        for param in self.model_k.parameters():
            param.requries_grad=False

        for epoch_id in range(epochs):
            loss_sum = 0
            tkb.set_description(desc="Batch Process Epoch " + str(epoch_id))
            for batch_id, data in enumerate(train_loader):
                if data[0].shape[0] != train_loader.batch_size:
                    continue
                optimizer.zero_grad()

                loss = self._forward(data)

                loss_sum += loss.item()
                tkb.set_postfix(loss='{:3f}'.format(
                    loss_sum / (batch_id+1)))
                tkb.update(1)

                self._update(loss=loss, optimizer=optimizer,
                             lr_scheduler=lr_scheduler)
                # There should be a better way...
                with torch.no_grad():
                    for param_q, param_k in zip(self.model.parameters(), self.model_k.parameters()):
                        param_k.data = param_k.data * self.m + \
                            param_q.data * (1. - self.m)
            tkb.reset()

    def _forward(self, data):
        imgs1, imgs2 = data

        q = self.model(imgs1.to('cuda'))
        with torch.no_grad():
            k = self.model_k(imgs2.to('cuda'))
            k = F.normalize(k)
        q = F.normalize(q)

        l_pos_1 = torch.bmm(q.view(q.shape[0], 1, q.shape[1]), k.view(
            k.shape[0], k.shape[1], 1)).squeeze().mean()

        
        q = self.model(imgs2.to('cuda'))
        with torch.no_grad():
            k = self.model_k(imgs1.to('cuda'))
            k = F.normalize(k)
        q = F.normalize(q)
        
        l_pos_2 = torch.bmm(q.view(q.shape[0], 1, q.shape[1]), k.view(
            k.shape[0], k.shape[1], 1)).squeeze().mean()

        loss = - 2 * (l_pos_1 + l_pos_2)
        return loss


class InstanceDiscriminationSupervisor(Supervisor):
    def __init__(self, dataset=None, data_augmentation=lambda dataset : AugmentationIndexedDataset(dataset, transformations=ContrastivePredictiveCodingAugmentations), embedding_size=128, n=500, t=0.07,
                 backbone=None, predictor=None, loss=nn.CrossEntropyLoss(reduction='mean')):
        """Instance Discrimination https://arxiv.org/pdf/1805.01978.pdf.

        Args:
            dataset (torch.utils.data.Dataset): The dataset to train on.
            data_augmentation (lambda dataset: dataset, optional): Lambda function that returns a dataset with augmentations. Defaults to lambda dataset : AugmentationIndexedDataset(dataset, transformations=ContrastivePredictiveCodingAugmentations).
            embedding_size (int, optional): Size of predicted embeddings. Defaults to 128.
            n (int, optional): Number of negative examples per instance. Defaults to 500.
            t (float, optional): Temperature. Defaults to 0.07.
            backbone (torch.nn.Module, optional): Backbone network. Defaults to None, resulting in an EfficientNet backbone.
            predictor (torch.nn.Module, optional): Prediction network. Defaults to None, resulting in a MLP that fits to the embeddings size.
            loss ([type], optional): The critierion to train on. Defaults to nn.CrossEntropyLoss(reduction='mean').
        """
        super().__init__(CombinedNet(ReshapeChannels(EfficientFeatures(norm_type='layer'))
                                     if backbone is None else backbone,
                                     Classification(
                                         layers=[3136, 1024, 1024, embedding_size])
                                     if predictor is None else predictor),
                         data_augmentation(dataset),
                         loss)
        self.embedding_size = embedding_size
        self.n = n
        self.t = t
        self.memory = BatchedMemory(size=len(self.dataset), embedding_size=self.embedding_size)

    def _epochs(self, epochs, train_loader, optimizer, lr_scheduler):
        batch_size = train_loader.batch_size
        # Init queue
        tkb = tqdm(total=int(len(train_loader)), bar_format="{l_bar}%s{bar}%s{r_bar}" % (
            Fore.GREEN, Fore.RESET), mininterval=0)
        for epoch_id in range(epochs):
            loss_sum = 0
            tkb.set_description(desc="Batch Process Epoch " + str(epoch_id))
            for batch_id, data in enumerate(train_loader):
                if data[0].shape[0] != train_loader.batch_size:
                    continue
                optimizer.zero_grad()

                loss = self._forward(data)
                loss_sum += loss.item()
                tkb.set_postfix(loss='{:3f}'.format(
                    loss_sum / (batch_id+1)))
                tkb.update(1)

                self._update(loss=loss, optimizer=optimizer,
                             lr_scheduler=lr_scheduler)
            tkb.reset()

    def _forward(self, data):
        imgs1, _, idx = data
        batch_size = imgs1.shape[0]

        q = self.model(imgs1.to('cuda'))
        k = self.memory.memory[idx]
        q = F.normalize(q)
        k = F.normalize(k)

        l_pos = torch.bmm(q.view(q.shape[0], 1, q.shape[1]), k.view(
            k.shape[0], k.shape[1], 1)).squeeze(1) / self.t
        l_neg = torch.bmm(q.view(q.shape[0], 1, q.shape[1]), F.normalize(self.memory.data(
            self.n, batch_size, idx)).permute(0, 2, 1)).squeeze(1) / self.t

        logits = torch.cat([l_pos, l_neg], dim=1)
        labels = torch.zeros(batch_size, device='cuda').long()
        loss = self.loss(logits, labels) + 0.01 * torch.mean(torch.norm(q - k, dim=1))

        with torch.no_grad():
            self.memory.update(q, idx)

        return loss


class ContrastiveMultiviewCodingSupervisor(Supervisor):
    def __init__(self, dataset=None,  data_augmentation=lambda dataset : AugmentationLabIndexedDataset(dataset, transformations=ContrastivePredictiveCodingAugmentations), embedding_size=128, n=3136, t=0.07, memory_m=0.5,
                 backbone=None, predictor=None, loss=nn.CrossEntropyLoss(reduction='mean')):
        """Contrastive Multiview Coding https://arxiv.org/pdf/1906.05849.pdf.

        Args:
            dataset (torch.utils.data.Dataset): The dataset to train on.
            data_augmentation (lambda dataset: dataset, optional): Lambda function that returns a dataset with augmentations. Defaults to lambda dataset : AugmentationLabIndexedDataset(dataset, transformations=ContrastivePredictiveCodingAugmentations).
            embedding_size (int, optional): Size of predicted embeddings. Defaults to 128.
            n (int, optional): Number of negatives. Defaults to 3136.
            t (float, optional): Temperature. Defaults to 0.07.
            memory_m (float, optional): Memory update momentum. Defaults to 0.5.
            backbone (torch.nn.Module, optional): Backbone network. Defaults to None, resulting in an EfficientNet backbone.
            predictor (torch.nn.Module, optional): Prediction network. Defaults to None, resulting in a MLP that fits to the embeddings size.
            loss ([type], optional): The critierion to train on. Defaults to nn.CrossEntropyLoss(reduction='mean').
        """
        super().__init__(CombinedNet(nn.Sequential(nn.Conv2d(1, 3, 1), ReshapeChannels(EfficientFeatures()))
                                     if backbone is None else backbone,
                                     Classification(
                                         layers=[3136, 1024, 1024, embedding_size])
                                     if predictor is None else predictor),
                         data_augmentation(dataset),
                         loss)
        self.embedding_size = embedding_size
        self.n = n
        self.t = t
        self.memory = BatchedMemory(size=len(self.dataset),
                               embedding_size=self.embedding_size, momentum=memory_m)

    def _epochs(self, epochs, train_loader, optimizer, lr_scheduler):
        tkb = tqdm(total=int(len(train_loader)), bar_format="{l_bar}%s{bar}%s{r_bar}" % (
            Fore.GREEN, Fore.RESET), mininterval=0)
        # Init queue
        self.model_k = nn.DataParallel(CombinedNet(nn.Sequential(nn.Conv2d(2, 3, 1), ReshapeChannels(EfficientFeatures())),
                                   Classification(layers=[3136, 1024, 1024, self.embedding_size])).to('cuda'))
        for epoch_id in range(epochs):
            loss_sum = 0
            tkb.set_description(desc="Batch Process Epoch " + str(epoch_id))
            for batch_id, data in enumerate(train_loader):
                if data[0].shape[0] != train_loader.batch_size:
                    continue
                optimizer.zero_grad()

                loss = self._forward(data)

                loss_sum += loss.item()
                tkb.set_postfix(loss='{:3f}'.format(
                    loss_sum / (batch_id+1)))
                tkb.update(1)

                self._update(loss=loss, optimizer=optimizer,
                             lr_scheduler=lr_scheduler)
            tkb.reset()

    def _forward(self, data):
        imgs1_l, imgs1_ab, imgs2_l, imgs2_ab, idx = data
        batch_size = imgs1_l.shape[0]

        # One way
        q1 = self.model(imgs1_l.to('cuda'))
        k1 = self.model_k(imgs2_ab.to('cuda'))
        q1 = F.normalize(q1)
        k1 = F.normalize(k1)

        l_pos1 = torch.bmm(q1.view(q1.shape[0], 1, q1.shape[1]), k1.view(
            k1.shape[0], k1.shape[1], 1)).squeeze(1) / self.t
        l_neg1 = torch.bmm(q1.view(q1.shape[0], 1, q1.shape[1]), self.memory.data(
            self.n, batch_size).permute(0, 2, 1)).squeeze(1) / self.t
        logits1 = torch.cat([l_pos1, l_neg1], dim=1)

        # The other
        q2 = self.model(imgs2_l.to('cuda'))
        k2 = self.model_k(imgs1_ab.to('cuda'))
        q2 = F.normalize(q2)
        k2 = F.normalize(k2)

        l_pos2 = torch.bmm(q2.view(q2.shape[0], 1, q2.shape[1]), k2.view(
            k2.shape[0], k2.shape[1], 1)).squeeze(1) / self.t
        l_neg2 = torch.bmm(q2.view(q1.shape[0], 1, q2.shape[1]), self.memory.data(
            self.n, batch_size).permute(0, 2, 1)).squeeze(1) / self.t
        logits2 = torch.cat([l_pos2, l_neg2], dim=1)

        logits = torch.cat((logits1, logits2), dim=0)
        labels = torch.zeros(batch_size * 2, device='cuda').long()
        loss = self.loss(logits, labels)

        with torch.no_grad():
            self.memory.update(k1, idx)

        return loss


class PIRLSupervisor(Supervisor):
    def __init__(self, dataset=None, data_augmentation=lambda dataset : AugmentationIndexedDataset(dataset, transformations=ContrastivePredictiveCodingAugmentations, transformations2=PIRLAugmentations), embedding_size=128, n=3136, t=0.07, memory_m=0.5,
                 backbone=None, predictor=None, loss=nn.CrossEntropyLoss(reduction='mean')):
        """PIRL https://arxiv.org/abs/1912.01991.

        Args:
            dataset (torch.utils.data.Dataset): The dataset to train on.
            data_augmentation (lambda dataset: dataset, optional): Lambda function that returns a dataset with augmentations. Defaults to lambda dataset : AugmentationIndexedDataset(dataset, transformations=ContrastivePredictiveCodingAugmentations, transformations2=PIRLAugmentations).
            embedding_size (int, optional): Size of predicted embeddings. Defaults to 128.
            n (int, optional): Number of negatives. Defaults to 3136.
            t (float, optional): Temperature. Defaults to 0.07.
            memory_m (float, optional): Memory update momentum. Defaults to 0.5.
            backbone (torch.nn.Module, optional): Backbone network. Defaults to None, resulting in an EfficientNet backbone.
            predictor (torch.nn.Module, optional): Prediction network. Defaults to None, resulting in a MLP that fits to the embeddings size.
            loss ([type], optional): The critierion to train on. Defaults to nn.CrossEntropyLoss(reduction='mean').
        """
        super().__init__(CombinedNet(ReshapeChannels(EfficientFeatures())
                                     if backbone is None else backbone,
                                     Classification(
                                         layers=[3136, 1024, 1024, embedding_size])
                                     if predictor is None else predictor),
                         data_augmentation(dataset),
                         loss)
        self.embedding_size = embedding_size
        self.n = n
        self.t = t
        self.memory = BatchedMemory(size=len(self.dataset),
                               embedding_size=self.embedding_size, momentum=memory_m)

    def _epochs(self, epochs, train_loader, optimizer, lr_scheduler):
        tkb = tqdm(total=int(len(train_loader)), bar_format="{l_bar}%s{bar}%s{r_bar}" % (
            Fore.GREEN, Fore.RESET), mininterval=0)
        # Init queue
        self.g_head = nn.DataParallel(CombinedNet(self.model.module.backbone, copy.deepcopy(self.model.module.predictor)))

        for epoch_id in range(epochs):
            loss_sum = 0
            tkb.set_description(desc="Batch Process Epoch " + str(epoch_id))
            for batch_id, data in enumerate(train_loader):
                if data[0].shape[0] != train_loader.batch_size:
                    continue
                optimizer.zero_grad()

                loss = self._forward(data,)

                loss_sum += loss.item()

                tkb.set_postfix(loss='{:3f}'.format(
                    loss_sum / (batch_id+1)))
                tkb.update(1)

                self._update(loss=loss, optimizer=optimizer,
                             lr_scheduler=lr_scheduler)
            tkb.reset()

    def _forward(self, data):
        imgs1, imgs2, idx = data
        batch_size = imgs1.shape[0]

        f = self.model(imgs1.to('cuda'))
        g = self.g_head(imgs2.to('cuda'))
        f = F.normalize(f)
        g = F.normalize(g)

        k = self.memory[idx]

        # f
        l_pos = torch.bmm(f.view(f.shape[0], 1, f.shape[1]), k.view(
            k.shape[0], k.shape[1], 1)).squeeze(1) / self.t
        l_neg = torch.bmm(f.view(f.shape[0], 1, f.shape[1]), self.memory.data(
            self.n, batch_size).permute(0, 2, 1)).squeeze(1) / self.t

        logits_f = torch.cat([l_pos, l_neg], dim=1)

        # g
        l_pos = torch.bmm(g.view(g.shape[0], 1, g.shape[1]), k.view(
            k.shape[0], k.shape[1], 1)).squeeze(1) / self.t
        l_neg = torch.bmm(g.view(g.shape[0], 1, g.shape[1]), self.memory.data(
            self.n, batch_size).permute(0, 2, 1)).squeeze(1) / self.t

        logits_g = torch.cat([l_pos, l_neg], dim=1)

        logits = torch.cat((logits_f, logits_g), dim=0)
        labels = torch.zeros(batch_size * 2, device='cuda').long()
        loss = self.loss(logits, labels)

        with torch.no_grad():
            self.memory.update(f, idx)

        return loss
