import torch
from torch import nn
from torch.nn import functional as F
from data import RotateDataset
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
                  num_workers=0, name="store/base", pretrained=False):
        print(bcolors.OKGREEN + "Train with " +
              type(self).__name__ + bcolors.ENDC)
        try:
            if pretrained:
                self.load(name)
        except Exception:
            raise IOError("Could not load CombinedNet.")
        try:
            train_loader = torch.utils.data.DataLoader(self.dataset, batch_size=batch_size,
                                                       shuffle=shuffle, num_workers=num_workers)
            optimizer = optimizer(self.model.parameters(), lr=lr)
            for epoch_id in range(epochs):
                tkb = tqdm(total=int(len(train_loader)), bar_format="{l_bar}%s{bar}%s{r_bar}" % (
                    Fore.GREEN, Fore.RESET), desc="Batch Process Epoch " + str(epoch_id))

                for data in train_loader:
                    inputs, labels = data
                    optimizer.zero_grad()

                    outputs = self.model(inputs.to('cuda'))
                    loss = self.loss(outputs, labels.to('cuda'))
                    loss.backward()
                    optimizer.step()
                    tkb.set_postfix(loss='{:3f}'.format(loss.item()))
                    tkb.update(1)

        except Exception as e:
            print(e)
        finally:
            self.save(name)
            print()

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
    def __init__(self, model, dataset, loss=nn.CrossEntropyLoss(reduction='mean')):
        super().__init__(model, RotateDataset(dataset), loss)
