# Super Selfish
A framework for self supervised learning on images.

Currently supports:
- ExemplarNet https://arxiv.org/abs/1406.6909
- RotateNet https://arxiv.org/abs/1803.07728 
  (No Context Free Network for performance reasons)
- Jigsaw Puzzle https://arxiv.org/abs/1603.09246
- Denoising Autoencoder https://www.cs.toronto.edu/~larocheh/publications/icml-2008-denoising-autoencoders.pdf
- Context Autencoder https://arxiv.org/pdf/1604.07379.pdf 
- BiGAN https://arxiv.org/pdf/1605.09782.pdf

Training is as easy as:

```python

supervisor = RotateNetSupervisor(train_dataset).to(device)
#supervisor = ExemplarNetSupervisor(train_dataset).to(device)
#supervisor = JigsawNetSupervisor(train_dataset).to(device)
#supervisor = DenoiseNetSupervisor(train_dataset).to(device)
#supervisor = ContextNetSupervisor(train_dataset).to(device)
#supervisor = BiGanSupervisor(train_dataset).to(device)

# Start training
supervisor.supervise(lr=lr, epochs=epochs,
                     batch_size=batch_size, name="store/base_" + supervisor_name, pretrained=False)

```



# TODOs
- DataParallel