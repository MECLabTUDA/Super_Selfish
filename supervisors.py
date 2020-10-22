import torch
from torch import nn
from torch.nn import functional as F
from data import visualize, RotateDataset, ExemplarDataset, JigsawDataset, DenoiseDataset, \
    ContextDataset, BiDataset, SplitBrainDataset, ContrastivePreditiveCodingDataset, AugmentationDataset, \
    AugmentationIndexedDataset, AugmentationLabIndexedDataset
from models import ReshapeChannels, Classification, Batch2Image, GroupedCrossEntropyLoss, \
    CPCLoss, MaskedCNN, EfficientFeatures, CombinedNet, Upsampling, ChannelwiseFC, GroupedEfficientFeatures, \
    GroupedUpsampling
from tqdm import tqdm
from colorama import Fore
from utils import bcolors
import numpy as np
import copy
from data import siamese_collate
from data import ContrastivePredictiveCodingAugmentations, MomentumContrastAugmentations
from memory import BatchedQueue, BatchedMemory


class Supervisor():
    def __init__(self, model, dataset, loss=nn.CrossEntropyLoss(reduction='mean'), collate_fn=None, distributed=False):
        self.model = nn.DataParallel(model) if distributed else model
        self.dataset = dataset
        self.loss = loss
        self.collate_fn = collate_fn

    def supervise(self, lr=1e-3, optimizer=torch.optim.Adam, epochs=10, batch_size=32, shuffle=True,
                  num_workers=0, name="store/base", pretrained=False, lr_scheduler=lambda optimizer: torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=1.0)):
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
        try:
            if pretrained:
                self.load(name)
        except Exception:
            raise IOError("Could not load pretrained.")

    def _init_data_optimizer(self, optimizer, batch_size, shuffle, num_workers, collate_fn, lr, lr_scheduler):
        train_loader = torch.utils.data.DataLoader(self.dataset, batch_size=batch_size,
                                                   shuffle=shuffle, num_workers=num_workers, collate_fn=collate_fn)
        optimizer = optimizer(self.model.parameters(), lr=lr)
        lr_scheduler = lr_scheduler(optimizer)

        return train_loader, optimizer, lr_scheduler

    def _epochs(self, epochs, train_loader, optimizer, lr_scheduler):
        for epoch_id in range(epochs):
            loss_sum = 0
            tkb = tqdm(total=int(len(train_loader)), bar_format="{l_bar}%s{bar}%s{r_bar}" % (
                Fore.GREEN, Fore.RESET), desc="Batch Process Epoch " + str(epoch_id))
            for batch_id, data in enumerate(train_loader):
                optimizer.zero_grad()

                loss = self._forward(data)
                loss_sum += loss.item()
                tkb.set_postfix(loss='{:3f}'.format(
                    loss_sum / (batch_id+1)))
                tkb.update(1)

                self._update(loss=loss, optimizer=optimizer,
                             lr_scheduler=lr_scheduler)

    def _forward(self, data):
        inputs, labels = data
        outputs = self.model(inputs.to('cuda'))
        loss = self.loss(outputs, labels.to('cuda'))
        return loss

    def _update(self, loss, optimizer, lr_scheduler):
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

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
                                         layers=[3136, 1024, 256, len(rotations)])
                                     if predictor is None else predictor),
                         RotateDataset(dataset, rotations=rotations),
                         loss)


class ExemplarNetSupervisor(Supervisor):
    def __init__(self, dataset, transformations=['rotation', 'crop', 'gray', 'flip', 'erase'], n_classes=8000, n_trans=100, max_elms=10, p=0.5,
                 backbone=None, predictor=None, loss=nn.CrossEntropyLoss(reduction='mean')):
        super().__init__(CombinedNet(ReshapeChannels(EfficientFeatures())
                                     if backbone is None else backbone,
                                     Classification(
                                         layers=[3136, 1024, 1024, n_classes])
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
                                         layers=[3136, 1024, 1024, 1000])
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
                             layers=[3136,  1024, 256, 1])) if discriminator is None else discriminator,
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
                             layers=[3136, 1024, 256, 1])) if discriminator is None else discriminator,
                         BiDataset(
                             dataset, shape=shape, rand_gen=rand_gen),
                         loss,
                         fake_loss)

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
    # Not the CFN of the paper for easier implementation with common backbones and, most importantly, easier reuse
    def __init__(self, dataset, half_crop_size=(int(28), int(28)), sides=['top', 'bottom', 'left', 'right'],
                 backbone=None, predictor=None, loss=CPCLoss(k=3, ignore=2).to('cuda'), collate_fn=siamese_collate):
        super().__init__(CombinedNet(Batch2Image(EfficientFeatures(norm_type='layer'))
                                     if backbone is None else backbone,
                                     nn.ModuleDict({side: ReshapeChannels(MaskedCNN(
                                         layers=[1280, 512, 256, 128, 128], mask=torch.from_numpy(np.array([[1, 1, 1], [1, 1, 1], [0, 0, 0]])),  side=side),
                                         in_channels=128, out_channels=64 * loss.k, kernel_size=1, padding=0, activation=nn.Identity, flat=False) for side in sides})
                                     if predictor is None else predictor),
                         ContrastivePreditiveCodingDataset(
                             dataset, half_crop_size=half_crop_size),
                         loss,
                         collate_fn)
        self.sides = sides

    def _forward(self, data):
        inputs, _ = data
        encodings = self.model.backbone(inputs.to('cuda'))
        # encodings = F.layer_norm(encodings.permute(
        #    0, 2, 3, 1), encodings.shape[1:2]).permute(0, 3, 1, 2)
        # predictions = F.layer_norm(predictions.permute(
        #    0, 2, 3, 1), predictions.shape[1:2]).permute(0, 3, 1, 2)
        loss = 0
        for side in self.sides:
            predictions = self.model.predictor[side](encodings)
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
    def __init__(self, dataset, transformations=['crop', 'gray', 'flip', 'jitter'], n_trans=10000, max_elms=3, p=0.5, embedding_size=64, K=8, m=0.999,  t=0.2,
                 backbone=None, predictor=None, loss=nn.CrossEntropyLoss(reduction='mean')):
        super().__init__(CombinedNet(ReshapeChannels(EfficientFeatures(norm_type='layer'))
                                     if backbone is None else backbone,
                                     Classification(
                                         layers=[3136, 1024, 1024, embedding_size])
                                     if predictor is None else predictor),
                         AugmentationDataset(
                             dataset, transformations=MomentumContrastAugmentations),
                         loss)
        self.embedding_size = embedding_size
        self.K = K
        self.m = m
        self.t = t

    def _epochs(self, epochs, train_loader, optimizer, lr_scheduler):
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
            tkb = tqdm(total=int(len(train_loader)), bar_format="{l_bar}%s{bar}%s{r_bar}" % (
                Fore.GREEN, Fore.RESET), desc="Batch Process Epoch " + str(epoch_id))
            for batch_id, data in enumerate(train_loader):
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
    def __init__(self, dataset, transformations=['crop', 'gray', 'flip', 'jitter'], n_trans=10000, max_elms=3, p=0.5, embedding_size=64, m=0.999,
                 backbone=None, predictor=None, loss=nn.CrossEntropyLoss(reduction='mean')):
        super().__init__(CombinedNet(CombinedNet(ReshapeChannels(EfficientFeatures()), Classification(layers=[3136, 1024, 1024, embedding_size]))
                                     if backbone is None else backbone,
                                     Classification(
                                         layers=[embedding_size, embedding_size * 4, embedding_size * 2, embedding_size])
                                     if predictor is None else predictor),
                         AugmentationDataset(
                             dataset, transformations=transformations, n_trans=n_trans, max_elms=max_elms, p=p),
                         loss)
        self.embedding_size = embedding_size
        self.m = m

    def _epochs(self, epochs, train_loader, optimizer, lr_scheduler):
        for epoch_id in range(epochs):
            loss_sum = 0
            tkb = tqdm(total=int(len(train_loader)), bar_format="{l_bar}%s{bar}%s{r_bar}" % (
                Fore.GREEN, Fore.RESET), desc="Batch Process Epoch " + str(epoch_id))
            self.model_k = copy.deepcopy(self.model)
            for batch_id, data in enumerate(train_loader):
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

    def _forward(self, data):
        imgs1, imgs2 = data

        q = self.model(imgs1.to('cuda'))
        with torch.no_grad():
            k = self.model_k.backbone(imgs2.to('cuda'))
            k = F.normalize(k)
        q = F.normalize(q)

        l_pos_1 = torch.bmm(q.view(q.shape[0], 1, q.shape[1]), k.view(
            k.shape[0], k.shape[1], 1)).squeeze().mean()

        q = self.model(imgs2.to('cuda'))
        with torch.no_grad():
            k = self.model_k.backbone(imgs1.to('cuda'))
            k = F.normalize(k)
        q = F.normalize(q)

        l_pos_2 = torch.bmm(q.view(q.shape[0], 1, q.shape[1]), k.view(
            k.shape[0], k.shape[1], 1)).squeeze().mean()

        loss = -2 * (l_pos_1 + l_pos_2)
        return loss


class InstanceDiscriminationSupervisor(Supervisor):
    def __init__(self, dataset, transformations=['crop', 'gray', 'flip', 'jitter'], n_trans=10000, max_elms=3, p=0.5, embedding_size=128, m=3136, t=0.03,
                 backbone=None, predictor=None, loss=nn.CrossEntropyLoss(reduction='mean')):
        super().__init__(CombinedNet(ReshapeChannels(EfficientFeatures())
                                     if backbone is None else backbone,
                                     Classification(
                                         layers=[3136, 1024, 1024, embedding_size])
                                     if predictor is None else predictor),
                         AugmentationIndexedDataset(
                             dataset, transformations=ContrastivePredictiveCodingAugmentations),
                         loss)
        self.embedding_size = embedding_size
        self.m = m
        self.t = t

    def _epochs(self, epochs, train_loader, optimizer, lr_scheduler):
        batch_size = train_loader.batch_size
        # Init queue
        memory = BatchedMemory(size=len(self.dataset), batch_size=batch_size,
                               embedding_size=self.embedding_size)

        for epoch_id in range(epochs):
            loss_sum = 0
            tkb = tqdm(total=int(len(train_loader)), bar_format="{l_bar}%s{bar}%s{r_bar}" % (
                Fore.GREEN, Fore.RESET), desc="Batch Process Epoch " + str(epoch_id))
            for batch_id, data in enumerate(train_loader):
                optimizer.zero_grad()

                loss = self._forward(data, memory)

                loss_sum += loss.item()
                tkb.set_postfix(loss='{:3f}'.format(
                    loss_sum / (batch_id+1)))
                tkb.update(1)

                self._update(loss=loss, optimizer=optimizer,
                             lr_scheduler=lr_scheduler)

    def _forward(self, data, memory):
        imgs1, imgs2, idx = data
        batch_size = imgs1.shape[0]

        q = self.model(imgs1.to('cuda'))
        k = self.model(imgs2.to('cuda'))
        q = F.normalize(q)
        k = F.normalize(k)

        l_pos = torch.bmm(q.view(q.shape[0], 1, q.shape[1]), k.view(
            k.shape[0], k.shape[1], 1)).squeeze(1) / self.t
        l_neg = torch.bmm(q.view(q.shape[0], 1, q.shape[1]), memory.data(
            self.m).permute(0, 2, 1)).squeeze(1) / self.t

        logits = torch.cat([l_pos, l_neg], dim=1)

        labels = torch.zeros(batch_size, device='cuda').long()
        loss = self.loss(logits, labels)

        with torch.no_grad():
            memory.update(k, idx)

        return loss


class ContrastiveMultiviewCodingSupervisor(Supervisor):
    def __init__(self, dataset, transformations=['crop', 'gray', 'flip', 'jitter'], n_trans=10000, max_elms=3, p=0.5, embedding_size=64, m=128, t=0.07,
                 backbone=None, predictor=None, loss=nn.CrossEntropyLoss(reduction='mean')):
        super().__init__(CombinedNet(nn.Sequential(nn.Conv2d(1, 3, 1), ReshapeChannels(EfficientFeatures()))
                                     if backbone is None else backbone,
                                     Classification(
                                         layers=[3136, 1024, 1024, embedding_size])
                                     if predictor is None else predictor),
                         AugmentationLabIndexedDataset(
                             dataset, transformations=ContrastivePredictiveCodingAugmentations),
                         loss)
        self.embedding_size = embedding_size
        self.m = m
        self.t = t

    def _epochs(self, epochs, train_loader, optimizer, lr_scheduler):
        batch_size = train_loader.batch_size
        # Init queue
        memory = BatchedMemory(size=len(self.dataset), batch_size=batch_size,
                               embedding_size=self.embedding_size, momentum=0.5)
        self.model_k = CombinedNet(nn.Sequential(nn.Conv2d(2, 3, 1), ReshapeChannels(EfficientFeatures())),
                                   Classification(layers=[3136, 1024, 1024, self.embedding_size])).to('cuda')
        for epoch_id in range(epochs):
            loss_sum = 0
            tkb = tqdm(total=int(len(train_loader)), bar_format="{l_bar}%s{bar}%s{r_bar}" % (
                Fore.GREEN, Fore.RESET), desc="Batch Process Epoch " + str(epoch_id))
            for batch_id, data in enumerate(train_loader):
                optimizer.zero_grad()

                loss = self._forward(data, memory)

                loss_sum += loss.item()
                tkb.set_postfix(loss='{:3f}'.format(
                    loss_sum / (batch_id+1)))
                tkb.update(1)

                self._update(loss=loss, optimizer=optimizer,
                             lr_scheduler=lr_scheduler)

    def _forward(self, data, memory):
        imgs1_l, imgs1_ab, imgs2_l, imgs2_ab, idx = data
        batch_size = imgs1_l.shape[0]

        # One way
        q1 = self.model(imgs1_l.to('cuda'))
        k1 = self.model_k(imgs2_ab.to('cuda'))
        q1 = F.normalize(q1)
        k1 = F.normalize(k1)

        l_pos1 = torch.bmm(q1.view(q1.shape[0], 1, q1.shape[1]), k1.view(
            k1.shape[0], k1.shape[1], 1)).squeeze(1) / self.t
        l_neg1 = torch.bmm(q1.view(q1.shape[0], 1, q1.shape[1]), memory.data(
            self.m).permute(0, 2, 1)).squeeze(1) / self.t
        logits1 = torch.cat([l_pos1, l_neg1], dim=1)

        # The other
        q2 = self.model(imgs2_l.to('cuda'))
        k2 = self.model_k(imgs1_ab.to('cuda'))
        q2 = F.normalize(q2)
        k2 = F.normalize(k2)

        l_pos2 = torch.bmm(q2.view(q2.shape[0], 1, q2.shape[1]), k2.view(
            k2.shape[0], k2.shape[1], 1)).squeeze(1) / self.t
        l_neg2 = torch.bmm(q2.view(q1.shape[0], 1, q2.shape[1]), memory.data(
            self.m).permute(0, 2, 1)).squeeze(1) / self.t
        logits2 = torch.cat([l_pos2, l_neg2], dim=1)

        logits = torch.cat((logits1, logits2), dim=0)
        labels = torch.zeros(batch_size * 2, device='cuda').long()
        loss = self.loss(logits, labels)

        with torch.no_grad():
            memory.update(k1, idx)

        return loss
