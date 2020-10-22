import torch
import numpy as np


class BatchedQueue():
    def __init__(self, K=8, batch_size=128, embedding_size=64, init_tensor=None):
        self.K = K
        self.batch_size = batch_size
        self.reset_pointer()
        if init_tensor is not None:
            self.queue = init_tensor
        else:
            self.queue = torch.randn(
                (self.K * batch_size, embedding_size), requires_grad=False, device='cuda')

    def reset_pointer(self):
        self.queue_pointer = 0

    def increment(self, step=1):
        self.queue_pointer = (self.queue_pointer + step) % self.K

    def init_w_loader_and_model(self, train_loader, model):
        with torch.no_grad():
            for batch_id, data in enumerate(train_loader):
                imgs1, imgs2 = data
                k = model(imgs2.to('cuda'))

                self.queue[self.queue_pointer *
                           self.batch_size:(self.queue_pointer + 1) * self.batch_size, :] = k

                if self.queue_pointer == self.K - 1:
                    break
                self.increment()

    def enqueue(self, k):
        self.queue[self.queue_pointer *
                   self.batch_size:(self.queue_pointer + 1) * self.batch_size, :] = k
        self.increment()

    def data(self):
        return self.queue


class BatchedMemory():
    def __init__(self, size=128, batch_size=128, embedding_size=64, init_tensor=None, momentum=1.0):
        self.size = size
        self.embedding_size = embedding_size
        self.batch_size = batch_size
        self.momentum = momentum
        if init_tensor is not None:
            self.memory = init_tensor
        else:
            self.memory = torch.randn(
                (self.size, embedding_size), requires_grad=False, device='cuda')

    def update(self, k, idx):
        self.memory[idx] = k * self.momentum + \
            self.memory[idx] * (1-self.momentum)

    def data(self, m):
        idx = np.random.choice(self.size, m * self.batch_size)
        return self.memory[idx].reshape(self.batch_size, m, -1)
