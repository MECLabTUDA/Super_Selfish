from models import ReshapeFeatures, ClassificationModule, EfficientFeatures, CombinedNet
from efficientnet_pytorch import EfficientNet
import torchvision.datasets as datasets
from supervisors import LabelSupervisor, RotateNetSupervisor
from torchvision import transforms
from torch import nn
import torch
from torch.utils.data import random_split

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Configuration
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Choose supervisor
supervisor_name = 'rotate'
lr = 1e-3
epochs = 1
batch_size = 32
device = 'cuda'

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Modules and data
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# EfficientNet backbone
backbone = ReshapeFeatures(EfficientFeatures())
# MLP Predictor
predictor = ClassificationModule()
# Combine both as a classification machine
combined = CombinedNet(backbone, predictor).to(device)
# Start off with CIFAR
train_dataset, val_dataset = random_split(datasets.CIFAR10(root='./datasets/', train=True,
                                                           download=False,
                                                           transform=transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])),
                                          [45000, 5000],
                                          generator=torch.Generator().manual_seed(42))
test_dataset = datasets.CIFAR10(root='./datasets/', train=False,
                                download=False, transform=transforms.Resize((224, 224)))

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Self Supervision
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Create supervisor
if supervisor_name == 'rotate':
    supervisor = RotateNetSupervisor(combined, train_dataset)

# Start training
supervisor.supervise(lr=lr, epochs=epochs,
                     batch_size=batch_size, name="store/base_" + supervisor_name, pretrained=False)


# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Finetune with self supervised features
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Finetune on "right" target
backbone = supervisor.get_backbone()
predictor = ClassificationModule([3136, 1024, 256, 10])
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
backbone = ReshapeFeatures(EfficientFeatures())
predictor = ClassificationModule([3136, 1024, 256, 10])
combined = CombinedNet(backbone, predictor).to(device)

# Label supervisor without self-supervision
supervisor = LabelSupervisor(combined, val_dataset)
# Start training
supervisor.supervise(lr=lr, epochs=epochs,
                     batch_size=batch_size, name="store/clean_" + supervisor_name, pretrained=False)
