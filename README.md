# Super Selfish
A unified Pytorch framework for image-based self-supervised learning.

## Algorithms
Currently support of 13 algorithms that can be run in parallel on one node of GPUs:
### Patch-based
- ExemplarNet https://arxiv.org/abs/1406.6909  
  We use the stronger set of augmentations used in CPC and do not use gradient-based patch sampling as this does not seem to be neccessary.
  We always process full images but apply scaling and translation.
- RotateNet https://arxiv.org/abs/1803.07728 
- Jigsaw Puzzle https://arxiv.org/abs/1603.09246  
  We apply random cropping within each patch to avoid border signals.  
  3x3 jigsaw too complicated for easy dataset, per default 2x2.  
  Jigsaw processed at once for performance and simplicity.
### Predictive
- Denoising Autoencoder https://www.cs.toronto.edu/~larocheh/publications/icml-2008-denoising-autoencoders.pdf
- Context Autencoder https://arxiv.org/pdf/1604.07379.pdf  
  We use the Random Block technique that randomly erases multiple small rectangles.
- SplitBrain Autoencoder https://arxiv.org/pdf/1611.09842.pdf  
  We use the classification architecture but do not restrict the predictiction to in gamut values.
### Generative
- BiGAN https://arxiv.org/pdf/1605.09782.pdf
### Contrastive
- Instance Discrimination https://arxiv.org/pdf/1805.01978.pdf  
  (Memory Bank, We made it Augmentation Task with CPC Augs, Only Projection head, 1 Backbone, Temperature)
- Contrastive Predictive Coding (V2) https://arxiv.org/pdf/1905.09272.pdf  
  (Batchwise, Future Prediction Task with augmentation, Target and Projection head, 1 Backbone, No Temperature)
- Momentum Contrast (V2) https://arxiv.org/pdf/2003.04297.pdf  
  (Queue, Augmentation Task, Projection Head, 1 Backbone and Momentum Encoder, Temperature)
  LayerNorm instead of ShuffledBN (on todo list)
- Contrastive Multiview Coding https://arxiv.org/pdf/1906.05849.pdf  
  (Memory Bank, Augmentation Task (We use CPC Aufs), Multimodal,Target and Projection head, 2 Backbones, No Temperature)
  Features Only from L channel as in theory, the embeddings should be close anyway
- Boostrap Your Own Latent (CL via BN) https://arxiv.org/pdf/2006.07733.pdf  
  (No negatives, Augmentation task, Target and Projection head, 2 Backbones,No Temperature)
- PIRL https://arxiv.org/abs/1912.01991  
  (Memory Bank, Augmentation + Jigsaw Task, Target and Projection Head, 1 Backbone, Temperature)  
  Jigsaw processed at once for performance and simplicity.

## Usage
### Requirements
Tested with  
CUDA 11.0 and Ubuntu 18.04  
torch 1.7.0 torchvision 0.8.1   
scikit-image 0.17.2  
elasticdeform 0.4.6  
tqdm 4.51.0  
scipy 1.5.4  
colorama 0.4.4  

Per default Super Selfish stores network parameters in the folder "store" in your directory and looks for a "dataset" folder.

### Install
```python
pip install super_selfish
```

### Training
For usage examples of all algorithms see test.py file.  
Be aware that pretext difficulty has to be adapted to your task and dataset.  
Further, contrastive methods mostly rely on enourmus batch sizes and mostly need a Multi-GPU setup.
Momentum Contrast typically also works with small batch sizes due to the queued structure.
<br><br>
Training is as easy as:
```python

# Choose supervisor
supervisor = RotateNetSupervisor(train_dataset) # .to('cuda')

# supervisor = RotateNetSupervisor(train_dataset)
# supervisor = ExemplarNetSupervisor(train_dataset)
# supervisor = JigsawNetSupervisor(train_dataset)
# supervisor = DenoiseNetSupervisor(train_dataset)
# supervisor = ContextNetSupervisor(train_dataset)
# supervisor = BiGanSupervisor(train_dataset)
# supervisor = SplitBrainNetSupervisor(train_dataset)
# supervisor = ContrastivePredictiveCodingSupervisor(train_dataset)
# supervisor = MomentumContrastSupervisor(train_dataset)
# supervisor = BYOLSupervisor(train_dataset)
# supervisor = InstanceDiscriminationSupervisor(train_dataset)
# supervisor = ContrastiveMultiviewCodingSupervisor(train_dataset)
# supervisor = PIRLSupervisor(train_dataset)

# Start training
supervisor.supervise(lr=1e-3, epochs=50,
                     batch_size=64, name="store/base", pretrained=False)

```
### Feature Extraction and Transfer
The model is automatically stored if the training ends after the given number of epochs or the user manualy interrupts the training process.  
If not directly reused in the same run, any model can be loaded with:

```python
supervisor = RotateNetSupervisor().load(name="store/base")
```
The feature extractor is retrieved using:
```python
# Returns the backbone network i.e. nn.Module
backbone_network = supervisor.get_backbone()
```
If you want to easily add new prediction head you can create a CombinedNet:
```python
CombinedNet(backbone_network, nn.Module(...)) 
```

### Flexibility
Although training is as easy as writing two lines of code, Super Selfish provides maximum flexibility. Any supervisor can be directly initialized with the corresponding hyperparameters. By default, the hyperparameters from the respective paper are used. Similiarily, the backbone architecture as well as prediction heads are by default those of the papers but can be customized as follows:
```python
supervisor = RotateNetSupervisor(train_dataset, backbone=nn.Module(...), predictor=nn.Module(...)) # .to('cuda')
```
For individual parameters see Algorithms.  

The training can be governed by the learning rate, the used optimizer, the batch size, wether to shuffle training data, and a learning rate schedule. Polyak averaging is soon to be added.
```python
def supervise(self, lr=1e-3, optimizer=torch.optim.Adam, epochs=10, batch_size=32, shuffle=True,
                  num_workers=0, name="store/base", pretrained=False, lr_scheduler=lambda optimizer: torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=1.0))
```
The supervise method of any Superviser is splitted into 5 parts such that functionalities can be easily updated/changed through overloading.
```python
# Loading of pretrained weights and models
def _load_pretrained(self, name, pretrained)
# Initialization of training specific objects
def _init_data_optimizer(self, optimizer, batch_size, shuffle, num_workers, collate_fn, lr, lr_scheduler)
# Wraps looping over epochs, batches. Takes care of visualizations and logging.
def _epochs(self, epochs, train_loader, optimizer, lr_scheduler)
# Implements one run of a model and other forward calculations
def _forward(self, data)
# Takes care of updating the modle, lr scheduler, ...
def _update(self, loss, optimizer, lr_scheduler)
```
The full documentation is available at: TODO

## Remarks
- If not precisley stated in a paper, we use the CPC image augmentations. Some augmentations or implementation details may be different to the original papers as we aim for a comparable unified framework.
- We use an EfficientNet https://github.com/lukemelas/EfficientNet-PyTorch implementation as the defaul backbone/feature extractor. We use a customized version that can be switched from batch norm to layer norm.
- Please feel free to open an issue regarding bugs and/or other algorithms that should be added.

## TODOs
- Multi node support, ShuffledBN
- Refactor old datasets, GANSupervisor
- Polyak Averaging
