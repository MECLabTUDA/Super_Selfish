from tqdm import tqdm
from colorama import Fore
import torch
import numpy as np


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def classification_loss(outputs, labels):
    _, predicted = torch.max(outputs.data, 1)
    total = labels.size(0)
    correct = (predicted == labels).sum()
    return correct / total


def test(model, test_dataset, loss_f=classification_loss, batch_size=48, shuffle=False, num_workers=0, collate_fn=None):
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                              shuffle=shuffle, num_workers=num_workers, collate_fn=collate_fn)

    tkb = tqdm(total=int(len(test_loader)), bar_format="{l_bar}%s{bar}%s{r_bar}" % (
        Fore.GREEN, Fore.RESET), desc="Test Accuracy ")

    loss_sum = 0
    for batch_id, data in enumerate(test_loader):
        if data[0].shape[0] != test_loader.batch_size:
            continue

        inputs, labels = data
        with torch.no_grad():
            outputs = model(inputs.to('cuda'))
            loss = loss_f(outputs, labels.to('cuda'))
            loss_sum += loss.item()
        tkb.set_postfix(Accuracy='{:3f}'.format(
            loss_sum / (batch_id+1)))
        tkb.update(1)
