from models import CombinedNet, Classification, ReshapeChannels, EfficientFeatures
from efficientnet_pytorch.model import EfficientNet
import torchvision.datasets as datasets
from supervisors import LabelSupervisor, RotateNetSupervisor, ExemplarNetSupervisor, \
    JigsawNetSupervisor, DenoiseNetSupervisor, ContextNetSupervisor, BiGanSupervisor, \
    SplitBrainNetSupervisor, ContrastivePredictiveCodingSupervisor, MomentumContrastSupervisor, \
    BYOLSupervisor, InstanceDiscriminationSupervisor, ContrastiveMultiviewCodingSupervisor
from torchvision import transforms
from torch import nn
import torch
from torch.utils.data import random_split

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Configuration
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Choose supervisor
supervisor_name = 'multiview'
lr = 1e-4
epochs = 50
batch_size = 32
device = 'cuda'

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Data
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Start off with CIFAR
train_dataset, val_dataset = random_split(datasets.CIFAR10(root='./datasets/', train=True,
                                                           download=False,
                                                           transform=transforms.Compose([transforms.Resize((225, 225)), transforms.ToTensor()])),
                                          [45000, 5000],
                                          generator=torch.Generator().manual_seed(42))
test_dataset = datasets.CIFAR10(root='./datasets/', train=False,
                                download=False, transform=transforms.Resize((225, 225)))
collate_fn = None
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Self Supervision
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Create supervisor
if supervisor_name == 'rotate':
    supervisor = RotateNetSupervisor(train_dataset).to(device)
elif supervisor_name == 'exemplar':
    supervisor = ExemplarNetSupervisor(train_dataset).to(device)
elif supervisor_name == 'jigsaw':
    supervisor = JigsawNetSupervisor(train_dataset).to(device)
elif supervisor_name == 'upsampling':
    supervisor = DenoiseNetSupervisor(train_dataset).to(device)
elif supervisor_name == 'context':
    supervisor = ContextNetSupervisor(train_dataset).to(device)
elif supervisor_name == 'bi':
    supervisor = BiGanSupervisor(train_dataset).to(device)
elif supervisor_name == 'splitbrain':
    supervisor = SplitBrainNetSupervisor(train_dataset).to(device)
elif supervisor_name == 'coding':
    supervisor = ContrastivePredictiveCodingSupervisor(
        train_dataset).to(device)
elif supervisor_name == 'momentum':
    supervisor = MomentumContrastSupervisor(train_dataset).to(device)
elif supervisor_name == 'byol':
    supervisor = BYOLSupervisor(train_dataset).to(device)
elif supervisor_name == 'discrimination':
    supervisor = InstanceDiscriminationSupervisor(train_dataset).to(device)
elif supervisor_name == 'multiview':
    supervisor = ContrastiveMultiviewCodingSupervisor(train_dataset).to(device)

# Start training
supervisor.supervise(lr=lr, epochs=epochs,
                     batch_size=batch_size, name="store/base_" + supervisor_name, pretrained=False)


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Finetune with self supervised features
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Finetune on "right" target
backbone = supervisor.get_backbone()
predictor = Classification([4096, 1024, 256, 10])
combined = CombinedNet(backbone, predictor).to(device)

# Label supervisor without self-supervision
supervisor = LabelSupervisor(combined, val_dataset)
# Start training
supervisor.supervise(lr=lr, epochs=epochs,
                     batch_size=batch_size, name="store/finetuned_" + supervisor_name, pretrained=False)

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Train clean pretrained EfficientNet
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Train on "right" target
backbone = ReshapeChannels(EfficientFeatures())
predictor = Classification([4096, 1024, 256, 10])
combined = CombinedNet(backbone, predictor).to(device)

# Label supervisor without self-supervision
supervisor = LabelSupervisor(combined, val_dataset)
# Start training
supervisor.supervise(lr=lr, epochs=epochs,
                     batch_size=batch_size, name="store/clean_" + supervisor_name, pretrained=False)
