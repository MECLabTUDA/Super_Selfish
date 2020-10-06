from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import numpy as np

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Library Datasets
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


class RotateDataset(Dataset):
    def __init__(self, dataset, rotations=[0.0, 90.0, 180.0,  -90.0]):
        self.dataset = dataset
        self.rotations = rotations

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        rot_id = int(np.random.randint(len(self.rotations), size=1))
        img = transforms.functional.to_pil_image(self.dataset[idx][0])
        img = transforms.functional.rotate(img, self.rotations[rot_id])
        img = transforms.functional.to_tensor(img)

        return img, rot_id
