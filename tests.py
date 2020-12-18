from super_selfish.models import CombinedNet, Classification, ReshapeChannels, EfficientFeatures
from super_selfish.efficientnet_pytorch.model import EfficientNet
import torchvision.datasets as datasets
from super_selfish.supervisors import LabelSupervisor, RotateNetSupervisor, ExemplarNetSupervisor, \
    JigsawNetSupervisor, DenoiseNetSupervisor, ContextNetSupervisor, BiGanSupervisor, \
    SplitBrainNetSupervisor, ContrastivePredictiveCodingSupervisor, MomentumContrastSupervisor, \
    BYOLSupervisor, InstanceDiscriminationSupervisor, ContrastiveMultiviewCodingSupervisor, \
    PIRLSupervisor
from torchvision import transforms
from torch import nn
import torch
from torch.utils.data import random_split, Subset
from super_selfish.utils import test
from super_selfish.data import LDataset
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Configuration
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Choose supervisor
supervisor_name = 'pirl'
lr = 1e-2
epochs = 2
batch_size = 32
device = 'cuda'
clean = False

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Data
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Start off with CIFAR

train_dataset = datasets.STL10(root='./datasets/', split='unlabeled',
                                                           download=False,
                                                           transform=transforms.Compose([transforms.Resize((225, 225)), transforms.ToTensor()]))
train_dataset = Subset(train_dataset, range(20000))
val_dataset = datasets.STL10(root='./datasets/', split='train',
                                                           download=False,
                                                           transform=transforms.Compose([transforms.Resize((225, 225)), transforms.ToTensor()]))
test_dataset = datasets.STL10(root='./datasets/', split='test',
                                download=False, transform=transforms.Compose([transforms.Resize((225, 225)), transforms.ToTensor()]))

#train_dataset = Subset(train_dataset, range(10000))
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
elif supervisor_name == 'denoise':
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
    val_dataset = LDataset(val_dataset)
    test_dataset = LDataset(test_dataset)
elif supervisor_name == 'pirl':
    supervisor = PIRLSupervisor(train_dataset).to(device)

if not clean:
    # Start training
    supervisor.supervise(lr=lr, epochs=epochs,
                         batch_size=batch_size, name="store/base_" + supervisor_name, pretrained=False)


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Finetune with self supervised features
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Finetune on "right" target with less epochs and lower lr
epochs = 1
lr = 1e-3

if clean:
    supervisor.load(name="store/base_" + supervisor_name)
combined = CombinedNet(supervisor.get_backbone(), Classification(
        layers=[3136 if supervisor_name != 'bi' else 3136 // 2, 10])).to(device)

# Label supervisor without self-supervision and only backprob through mlp
supervisor = LabelSupervisor(combined, val_dataset)
# Start training
supervisor.supervise(lr=lr, epochs=epochs,
                     batch_size=batch_size, name="store/finetuned_" + supervisor_name, pretrained=False)
test(combined, test_dataset)
