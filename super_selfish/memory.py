import torch
import numpy as np
from torch.utils.data import DataLoader

class BatchedQueue():
    def __init__(self, K=8, batch_size=128, embedding_size=64, init_tensor=None):
        """Queue that works on batches.

        Args:
            K (int, optional): Number of batches stored in queue. Defaults to 8.
            batch_size (int, optional): Size of a batch. Defaults to 128.
            embedding_size (int, optional): Size of stored instances. Defaults to 64.
            init_tensor (torch.FloatTensor, optional): Initializes queue with given tensor. Defaults to None.
        """
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
        """Increments queue pointer.

        Args:
            step (int, optional): Size of increment. Defaults to 1.
        """
        self.queue_pointer = (self.queue_pointer + step) % self.K

    def init_w_loader_and_model(self, train_loader, model):
        """Initializes queue with a given model and data. Expects data to be a tuple of two
            with the instance to process at the second position.

        Args:
            train_loader (torch.utils.data.DataLoader): Data to use for initialization. 
            model (torch.nn.Module): The module to use for initialization.
        """
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
        """ Updates queue at current queue position.

        Args:
            k (torch.FloatTensor): Tensor to insert.
        """
        self.queue[self.queue_pointer *
                   self.batch_size:(self.queue_pointer + 1) * self.batch_size, :] = k
        self.increment()

    def data(self):
        """Returns queue data.

        Returns:
            torch.FloatTensor: Queue data.
        """
        return self.queue


class BatchedMemory():
    def __init__(self, size=128, embedding_size=64, init_tensor=None, momentum=1.0):
        """Memory that works on batches.

        Args:
            size (int, optional): Number of memory entries. Defaults to 128.
            embedding_size (int, optional): Size of stored instances. Defaults to 64.
            init_tensor (torch.FloatTensor, optional): Initializes memorywith given tensor. Defaults to None.
            momentum (float, optional): Updates memory only partially, where 1.0 uses only the new representation whereas 0.0 uses only the old representation. Defaults to 1.0.
        """
        self.size = size
        self.embedding_size = embedding_size
        self.momentum = momentum
        if init_tensor is not None:
            self.memory = init_tensor
        else:
            self.memory = torch.randn(
                (self.size, embedding_size), requires_grad=False, device='cuda')

    def update(self, k, idx):
        """Updates memory at given position

        Args:
            k (torch.FloatTensor): Tensor to insert.
            idx (torch.LongTensor): Where to insert.
        """
        self.memory[idx] = k * self.momentum + \
            self.memory[idx] * (1-self.momentum)
    
    def load(self, name):
        self.memory = torch.load(name)
    
    def save(self, name):
        torch.save(self.memory, name)

    def data(self, m, batch_size, but_idx=None):
        """Returns m many random memory entries.

        Args:
            m (int): Number of entries to return.
            but_idx ([int], optional): Which indices to ignore. Defaults to None.

        Returns:
            torch.FloatTensor: Memory data.
        """
        idx = np.random.choice(np.delete(np.arange(self.size), but_idx) if but_idx is not None else np.arange(self.size), m * batch_size)
        return self.memory[idx].reshape(batch_size, m, -1)

    def __getitem__(self, idx):
        return self.memory[idx]
