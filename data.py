from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms, utils
import numpy as np
import torch
import random

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Library Datasets
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


class DenoiseDataset(Dataset):
    def __init__(self, dataset, p=0.7):
        self.dataset = dataset
        self.p = p

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img = self.dataset[idx][0]
        mask = np.random.rand(1, img.shape[1], img.shape[2]) >= self.p
        noised_img = img * mask.astype(int)
        return noised_img.float(), img.float()


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


class ExemplarDataset(Dataset):
    def __init__(self, dataset, transformations, n_classes=8000, n_trans=100, max_elms=10, p=0.1):
        # More memory efficient online implementation, even harder task
        # Processes full images, we are living in the twenties...
        # Thereby automatically capture various scales
        indices = torch.randint(len(dataset), (n_classes,)).long()
        self.dataset = Subset(dataset, indices)
        self.p = p
        self.n_trans = n_trans
        elm_transformations = []
        for t in transformations:
            if t == 'rotation':
                elm_transformations.append(transforms.RandomRotation(
                    20, resample=False, expand=False, center=None, fill=None))
            elif t == 'crop':
                elm_transformations.append(transforms.RandomResizedCrop(
                    dataset[0][0].shape[1:], scale=(0.5, 1.0), ratio=(0.75, 1.3333333333333333), interpolation=2))
            elif t == 'gray':
                elm_transformations.append(transforms.RandomGrayscale(p=1.0))
            elif t == 'flip':
                elm_transformations.append(
                    transforms.RandomHorizontalFlip(1.0))
            elif t == 'erase':
                elm_transformations.append(transforms.Compose(
                    [transforms.ToTensor(), transforms.RandomErasing(1.0), transforms.ToPILImage()]))
            else:
                elm_transformations.append(t)

        self.transformations = []
        for _ in range(self.n_trans):
            transformation = []
            for t in range(max_elms):
                if random.random() < self.p:
                    transformation.append(
                        transforms.RandomChoice(elm_transformations))
            self.transformations.append(transforms.Compose(transformation))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img = transforms.functional.to_pil_image(self.dataset[idx][0])
        img = transforms.RandomChoice(self.transformations)(img)
        img = transforms.functional.to_tensor(img)
        return img, idx


class JigsawDataset(Dataset):
    def __init__(self, dataset, jigsaw_path="utils/permutations_hamming_max_1000.npy", n_perms_per_image=69, crop_size=64):
        # More memory efficient online implementation, even harder task
        # Processes full images, we are living in the twenties...
        # Thereby automatically capture various scales
        self.dataset = dataset
        self.permutations = np.load(jigsaw_path)
        self.perms_per_image = np.random.choice(
            self.permutations.shape[0], len(dataset) * n_perms_per_image).reshape(len(dataset), n_perms_per_image)
        self.s = self.dataset[0][0].shape[1] // 3
        self.crop = transforms.Compose([transforms.RandomCrop(
            crop_size, pad_if_needed=True), transforms.Resize((self.s, self.s)), transforms.ColorJitter(0.1, 0.1, 0.1, 0.1)])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        perm_id = np.random.choice(self.perms_per_image[idx])
        perm = self.permutations[perm_id]
        # Looking for a cleaner and more beautifull way
        img = transforms.functional.to_pil_image(self.dataset[idx][0])
        img_out = img.copy()
        for n in range(9):
            i = (n // 3) * self.s
            j = (n % 3) * self.s

            patch = transforms.functional.to_tensor(
                self.crop(img.crop(box=(i, j, i + self.s, j + self.s))))
            patch_mean = torch.mean(patch)
            patch_std = torch.std(patch)
            patch_std = 1 if patch_std == 0 else patch_std
            normed_patch = transforms.functional.normalize(
                patch, patch_mean, patch_std)
            normed_patch = transforms.functional.to_pil_image(normed_patch)

            i_out = (perm[n] // 3) * self.s
            j_out = (perm[n] % 3) * self.s

            img_out.paste(normed_patch, box=(
                i_out, j_out, i_out + self.s, j_out + self.s))

        img_out = transforms.functional.to_tensor(img_out)
        return img_out, perm_id

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Data utils
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


def visualize(dataset, idx=0, folder_path='visualization/'):
    pil_img = transforms.functional.to_pil_image(dataset[0][0])
    pil_img.save(folder_path + type(dataset).__name__ +
                 "-" + str(idx) + ".png")
