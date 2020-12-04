from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms, utils
import numpy as np
from scipy.ndimage import gaussian_filter
import elasticdeform
import math
import torch
import random
from skimage.color import lab2rgb, rgb2lab
from PIL import Image as im_
from PIL import ImageOps as imo
from PIL import ImageEnhance as ime
from PIL import ImageFilter as imf

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Library Datasets
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


class AugmentationDataset(Dataset):
    def __init__(self, dataset, transformations, transformations2=None, clean1=False, clean2=False):
        """Standard dataset for contrastive algorithms. Augments each image with given transformations.

        Args:
            dataset (torch.utils.data.Dataset)): The backbone dataset.
            transformations (function(PIL.Image) identifier): Transformations for either img1 or img1 and img2. Must return a PIL.Image.
            transformations2 (function(PIL.Image) identifier, optional): Transformations for img2. Defaults to None. Must return a PIL.Image if specified.
            clean1 (bool, optional): Wether to not augment img1. Defaults to False.
            clean2 (bool, optional): Wether to not augment img2. Defaults to False.

        Returns:
            Tuple via __getitem__: Two augmentations of an image.
        """
        self.dataset = dataset
        self.trans = transformations
        self.trans2 = transformations2
        self.clean1 = clean1
        self.clean2 = clean2

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img = transforms.functional.to_pil_image(self.dataset[idx][0])

        if self.clean1:
            img1 = img
        else:
            img1 = self.trans(img)

        if self.clean2:
            img2 = img
        else:
            if self.trans2 is None:
                img2 = self.trans(img)
            else:
                img2 = self.trans2(img)

        img1 = transforms.functional.to_tensor(img1)
        img2 = transforms.functional.to_tensor(img2)

        return img1, img2

class LDataset(Dataset):
    def __init__(self, dataset):
        """Extracts the L channel of an RGB image and keeps the label the same.

        Args:
            dataset (torch.utils.data.Dataset)): The backbone dataset.
        """    
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img1, label = self.dataset[idx]
        img1 = rgb2lab(img1.permute(1, 2, 0).cpu().numpy())

        img1_l = torch.from_numpy(img1).permute(
            2, 0, 1)[0:1, :, :]

        return img1_l, label

class AugmentationIndexedDataset(AugmentationDataset):
    def __init__(self, dataset, transformations, transformations2=None, clean1=False, clean2=False):
        """Extends AugmentationDataset to return instance index.

        Args:
            dataset (torch.utils.data.Dataset)): The backbone dataset.
            transformations (function(PIL.Image)): Transformations for either img1 or img1 and img2. Must return a PIL.Image.
            transformations2 (function(PIL.Image), optional): Transformations for img2. Defaults to None. Must return a PIL.Image if specified.
            clean1 (bool, optional): Wether to not augment img1. Defaults to False.
            clean2 (bool, optional): Wether to not augment img2. Defaults to False.

        Returns:
            Tuple via __getitem__: Two augmentations of an image and the image index.
        """
        super().__init__(dataset, transformations, transformations2, clean1, clean2)

    def __getitem__(self, idx):
        img1, img2 = super().__getitem__(idx)
        return img1, img2, idx


class AugmentationLabIndexedDataset(AugmentationIndexedDataset):
    def __init__(self, dataset, transformations, transformations2=None, clean1=False, clean2=False):
        """Extends AugmentationIndexedDataset to split images into l and ab channels

        Args:
            dataset (torch.utils.data.Dataset)): The backbone dataset.
            transformations (function(PIL.Image)): Transformations for either img1 or img1 and img2. Must return a PIL.Image.
            transformations2 (function(PIL.Image), optional): Transformations for img2. Defaults to None. Must return a PIL.Image if specified.
            clean1 (bool, optional): Wether to not augment img1. Defaults to False.
            clean2 (bool, optional): Wether to not augment img2. Defaults to False.

        Returns:
            Tuple via __getitem__: Two augmentations of an image as l and ab channels as well as the image index.
        """
        super().__init__(dataset, transformations, transformations2, clean1, clean2)

    def __getitem__(self, idx):
        img1, img2, idx = super().__getitem__(idx)
        img1, img2 = rgb2lab(img1.permute(1, 2, 0).cpu().numpy()), rgb2lab(
            img2.permute(1, 2, 0).cpu().numpy())

        img1_l, img1_ab = torch.from_numpy(img1).permute(
            2, 0, 1)[0:1, :, :], torch.from_numpy(img1).permute(2, 0, 1)[1:, :, :]

        img2_l, img2_ab = torch.from_numpy(img2).permute(
            2, 0, 1)[0:1, :, :], torch.from_numpy(img2).permute(2, 0, 1)[1:, :, :]

        return img1_l, img1_ab, img2_l, img2_ab, idx


class ContrastivePreditiveCodingDataset(Dataset):
    def __init__(self, dataset, half_crop_size=(28, 28)):
        """CPC v2 dataset

        Args:
            dataset (torch.utils.data.Dataset)): The backbone dataset.
            half_crop_size (tuple, optional): Half size of crops to predict. Defaults to (int(28), int(28)).
        """
        self.dataset = dataset
        self.half_crop_size = half_crop_size

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img = self.dataset[idx][0]
        n_x = img.shape[1] // self.half_crop_size[0] - 1
        n_y = img.shape[2] // self.half_crop_size[1] - 1

        crops = []
        for i in range(n_x):
            for j in range(n_y):
                crop = img[:, i * self.half_crop_size[0]: (i+2) * self.half_crop_size[0],
                           j * self.half_crop_size[1]: (j+2) * self.half_crop_size[1]]
                crop = transforms.functional.to_pil_image(crop)
                crop = ContrastivePredictiveCodingAugmentations(crop)
                crop = transforms.functional.to_tensor(crop)
                crops.append(crop)

        crops = torch.stack(crops)
        return crops, crops


class SplitBrainDataset(Dataset):
    def __init__(self, dataset, l_step=2, l_offset=0, ab_step=26, a_offset=128, b_offset=128):
        """Splitbrain autoencoder dataset.

        Args:
            dataset (torch.utils.data.Dataset)): The backbone dataset.
            l_step (int, optional): l channel bin size. Defaults to 2.
            l_offset (int, optional): l channel offset. Defaults to 0.
            ab_step (int, optional): ab channel bin size Defaults to 26.
            a_offset (int, optional): a channel offset. Defaults to 128.
            b_offset (int, optional): b channel offset. Defaults to 128.
        """
        self.dataset = dataset
        # No gammut implementation
        self.l_step = l_step
        self.l_offset = l_offset
        self.ab_step = ab_step
        self.a_offset = a_offset
        self.b_offset = b_offset
        self.n_bins = b_offset * 2 // ab_step

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img = self.dataset[idx][0].permute(1, 2, 0).cpu().numpy()
        lab_img = torch.from_numpy(rgb2lab(img)).permute(2, 0, 1)

        lab_lab = lab_img.clone()
        lab_lab[0, :, :] += self.l_offset
        lab_lab[0, :, :] //= self.l_step
        lab_lab[0, :, :] = torch.max(torch.zeros(1), lab_lab[0, :, :])

        lab_lab[1, :, :] += self.a_offset
        lab_lab[2, :, :] += self.b_offset
        lab_lab[[1, 2], :, :] //= self.ab_step
        lab_lab[1, :, :] *= self.n_bins
        lab_lab[1, :, :] += lab_lab[2, :, :]
        lab_lab = lab_lab[: 2, :, :]

        return lab_img.float(), lab_lab.long()


class DenoiseDataset(Dataset):
    def __init__(self, dataset, p=0.7):
        """Denoising autoencoder dataset.
        Args:
            dataset (torch.utils.data.Dataset)): The backbone dataset.
            p (float, optional): Noise level. Defaults to 0.7.
        """
        self.dataset = dataset
        self.p = p

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img = self.dataset[idx][0]
        mask = np.random.rand(1, img.shape[1], img.shape[2]) >= self.p
        noised_img = img * mask.astype(int)
        return noised_img.float(), img.float()


class BiDataset(Dataset):
    def __init__(self, dataset, shape=(32, 7, 7), rand_gen=np.random.rand):
        """BiGAN dataset.

        Args:
            dataset (torch.utils.data.Dataset)): The backbone dataset.
            shape (tuple, optional): Latent vector shape. Defaults to (32, 8, 8).
            rand_gen (np.random, optional): Random noise distribution. Defaults to np.random.rand.
        """
        self.dataset = dataset
        self.rand_gen = rand_gen
        self.shape = shape

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img = self.dataset[idx][0]
        z = torch.from_numpy(self.rand_gen(*self.shape))
        return z.float(), img.float()


class ContextDataset(Dataset):
    def __init__(self, dataset, p=0.3, n_blocks=10, scale_range=(0.05, 0.1)):
        """ContextAutoencoder dataset.

        Args:
            dataset (torch.utils.data.Dataset)): The backbone dataset.
            p (float, optional): Prob. with which a block is erased. Defaults to 0.3.
            n_blocks (int, optional): Number of blocks that may be erased in a given image. Defaults to 10.
            scale_range (tuple, optional): Block scale range from which a scale is sampled per block. Defaults to (0.05, 0.1).
        """
        self.dataset = dataset
        self.n_blocks = n_blocks
        self.erase = transforms.RandomErasing(
            p=p, scale=scale_range, ratio=(0.3, 3.3), value=0, inplace=False)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img = self.dataset[idx][0]
        erased_img = img
        for _ in range(self.n_blocks):
            erased_img = self.erase(erased_img)

        return erased_img, img


class RotateDataset(Dataset):
    def __init__(self, dataset, rotations=[0.0, 90.0, 180.0,  -90.0], r_all=True):
        """RotateNet dataset.

        Args:
            dataset (torch.utils.data.Dataset)): The backbone dataset.
            rotations (list, optional): Rotations to predict. Defaults to [0.0, 90.0, 180.0,  -90.0].
            r_all (bool, optional): Wether to return all rotations at once. Defaults to True.

        """

        self.dataset = dataset
        self.rotations = rotations
        self.r_all = r_all

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if not self.r_all:
            rot_id = int(np.random.randint(len(self.rotations), size=1))
            img = transforms.functional.to_pil_image(self.dataset[idx][0])
            img = transforms.functional.rotate(img, self.rotations[rot_id])
            img = transforms.functional.to_tensor(img)

            return img, rot_id
        else:
            imgs = []
            img_c = transforms.functional.to_pil_image(self.dataset[idx][0])
            for rot_id, _ in enumerate(self.rotations):
                img = transforms.functional.rotate(
                    img_c, self.rotations[rot_id])
                img = transforms.functional.to_tensor(img)
                imgs.append(img)
            return torch.stack(imgs), torch.arange(len(self.rotations))


class ExemplarDataset(Dataset):
    def __init__(self, dataset, transformations=None, n_classes=8000, n_trans=100, max_elms=10, p=0.5):
        """ExemplarNet dataset.

        Args:
            dataset (torch.utils.data.Dataset): The dataset to train on.
            transformations (list, optional): Type of elementar transformations to use.
            n_classes (int, optional): Number of classes, i.e. the subset size of the dataset. Defaults to 8000.
            n_trans (int, optional): Number of combined transformations. Defaults to 100.
            max_elms (int, optional): Number of elementar transformations per combined transformation. Defaults to 10.
            p (float, optional): Prob. of an elmentar transformation to be part of a combined transformation. Defaults to 0.5.
        """
        pool = [transforms.RandomRotation(  # Rotation
            30, resample=False, expand=False, center=None, fill=None),
            transforms.RandomAffine(  # Shearing
            0, translate=None, scale=None, shear=30, resample=False, fillcolor=0),
            transforms.RandomAffine(  # Translate
            0, translate=(0.3, 0.3), scale=None, shear=None, resample=False, fillcolor=0),
            transforms.Lambda(lambda x: imo.autocontrast(x)),  # Autocontrast
            transforms.Lambda(lambda x: imo.invert(x)),  # Invert
            transforms.Lambda(lambda x: imo.equalize(x)),  # Equalize
            transforms.Lambda(lambda x: imo.solarize(x)),  # Solarize
            transforms.Lambda(lambda x: imo.posterize(
                x, bits=int(np.random.randint(4, 8) + 1))),  # Posterize
            transforms.Lambda(lambda x: ime.Color(
                x).enhance(np.random.uniform())),  # Color
            transforms.Lambda(lambda x: ime.Brightness(
                x).enhance(np.random.uniform())),  # Brightness
            transforms.Lambda(lambda x: ime.Contrast(
                x).enhance(np.random.uniform())),  # Contrast
            transforms.Lambda(lambda x: ime.Sharpness(
                x).enhance(np.random.uniform())),  # Sharpness
            transforms.Compose(  # Set black
            [transforms.ToTensor(), transforms.RandomErasing(1.0), transforms.ToPILImage()]),
            transforms.Lambda(lambda x: transforms.functional.to_grayscale(  # Grayscale
                x, num_output_channels=3)),
            transforms.Lambda(lambda x: elastic_transform(x, sigma=10))
        ]

        # Processes full images and apply random cropping instead of gradient based sampling.
        indices = torch.randint(len(dataset), (n_classes,)).long()
        self.dataset = Subset(dataset, indices)
        self.p = p
        self.n_trans = n_trans
        elm_transformations = transformations if transformations is not None else pool

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
    def __init__(self, dataset, transformations=lambda x: JigsawAugmentations(x), jigsaw_path="super_selfish/utils/permutations_hamming_max_1000.npy", n_perms_per_image=24, total_perms=24, crops=2, crop_size=112):
        """Jigsaw puzzle dataset.

        Args:
            dataset (torch.utils.data.Dataset)): The backbone dataset.
            transformations ([type], optional): Transformations in addition to random patch cropping. Defaults to ContrastivePredictiveCodingAugmentations.
            jigsaw_path (str, optional): The path to the used permutations. Defaults to "utils/permutations_hamming_max_1000.npy".
            n_perms_per_image (int, optional): Number of permutations per image. Defaults to 24.
            total_perms (int, optional): Number of perms in total. Defaults to 24.
            crops (int, optional): Number of patches is crops x crops. Defaults to 2.
            crop_size (int, optional): Crop size, implicitly determines the distance between crops. Defaults to 112.
        """
        self.dataset = dataset
        self.permutations = np.load(jigsaw_path)[:total_perms]
        # We fix the number of permutations per image
        self.perms_per_image = np.random.choice(
            self.permutations.shape[0], len(dataset) * n_perms_per_image).reshape(len(dataset), n_perms_per_image)
        self.s = self.dataset[0][0].shape[1] // crops
        self.crops = crops
        self.trans = transforms.Compose([transforms.RandomCrop(
            crop_size, pad_if_needed=True), transforms.Resize((self.s, self.s)),
            transforms.Lambda(transformations)])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        perm_id = np.random.choice(self.perms_per_image[idx])
        perm = self.permutations[perm_id]
        # Looking for a cleaner and more beautifull way
        img_out = jigsaw(transforms.functional.to_pil_image(
            self.dataset[idx][0]), perm, self.s, self.trans, crops=self.crops, normed=True)
        img_out = transforms.functional.to_tensor(img_out)
        return img_out, perm_id

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Data utils
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


def visualize(dataset, idx=0, folder_path='', batched=False):
    """Deprecated.
    """
    if batched:
        img = dataset[0][0][0]
    else:
        img = dataset[0][0]

    pil_img = transforms.functional.to_pil_image(img)
    pil_img.save(folder_path + type(dataset).__name__ +
                 "-" + str(idx) + ".png")


def batched_collate(data):
    transposed_data = list(zip(*data))
    img = torch.cat(transposed_data[0], 0)
    labels = torch.cat(transposed_data[1], 0)
    return img, labels


# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Transformations
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def jigsaw(img, perm, s, trans=lambda x: x, normed=True, crops=3):
    """Jigsaws image into crops and shuffles.

    Args:
        img (PIL.Image): Image.
        perm ([int]): Permutation to apply.
        s (int): Output size of each crop as well as size of area to crop from
        trans (function(PIL.Image) identifier, optional): Augmentations on crop. Defaults to lambdax:x.
        normed (bool, optional): Wether to norm crop. Defaults to True.

    Returns:
        PIL.Image: Output image
    """
    img_out = img.copy()
    for n in range(crops * crops):
        i = (n // crops) * s
        j = (n % crops) * s

        patch = transforms.functional.to_tensor(
            trans(img.crop(box=(i, j, i + s, j + s))))

        if normed:
            patch_mean = torch.mean(patch)
            patch_std = torch.std(patch)
            patch_std = 1 if patch_std == 0 else patch_std
            normed_patch = transforms.functional.normalize(
                patch, patch_mean, patch_std)
        else:
            normed_patch = patch

        normed_patch = transforms.functional.to_pil_image(normed_patch)

        i_out = (perm[n] // crops) * s
        j_out = (perm[n] % crops) * s

        img_out.paste(normed_patch, box=(
            i_out, j_out, i_out + s, j_out + s))
    return img_out


def elastic_transform(img, sigma):
    """Elastic 3x3 transform like for U-Nets.

    Args:
        img (PIL.Image): Image.
        sigma (float): Std. Dev.

    Returns:
        PIL.Image: Output image
    """
    def t1(image, sigma): return elasticdeform.deform_random_grid(
        image, axis=(0, 1), sigma=sigma)

    img = transforms.functional.to_tensor(img)
    img = torch.from_numpy(t1(img.permute(1, 2, 0).cpu(
    ).numpy(), sigma=10.0)).permute(2, 0, 1)
    img = transforms.functional.to_pil_image(img)
    return img


def MomentumContrastAugmentations(img):
    pool = transforms.Compose([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
                               transforms.RandomHorizontalFlip(p=0.5),
                               transforms.RandomGrayscale(p=0.2)])

    img = pool(img)
    # Guassian Blur
    if np.random.uniform() < 0.5:
        img = img.filter(imf.GaussianBlur(
            radius=np.random.uniform(0.1, 2.0)))

    return img


def BYOLAugmentations(img):

    t1 = transforms.Compose([transforms.RandomResizedCrop(img.size, scale=(
        0.08, 1.0), ratio=(0.75, 1.3333333333333333), interpolation=im_.BICUBIC),
        transforms.RandomHorizontalFlip(p=0.5)])
    img = t1(img)

    if np.random.uniform() < 0.8:
        t2 = transforms.ColorJitter(
            brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)
        img = t2(img)

    if np.random.uniform() < 0.2:
        img = transforms.functional.to_grayscale(img, num_output_channels=3)

    if np.random.uniform() < 0.3:
        img = img.filter(imf.GaussianBlur(
            radius=np.random.uniform(0.1, 2.0)))

    if np.random.uniform() < 0.2:
        t3 = transforms.Lambda(lambda x: imo.solarize(x))
        img = t3(img)

    return img


def ContrastivePredictiveCodingAugmentations(img):
    # We use transformations as traceable
    # https://arxiv.org/pdf/1805.09501.pdf
    pool = [transforms.RandomRotation(  # Rotation
        30, resample=False, expand=False, center=None, fill=None),
        transforms.RandomAffine(  # Shearing
        0, translate=None, scale=None, shear=30, resample=False, fillcolor=0),
        transforms.RandomAffine(  # Translate
        0, translate=(0.3, 0.3), scale=None, shear=None, resample=False, fillcolor=0),
        transforms.Lambda(lambda x: imo.autocontrast(x)),  # Autocontrast
        transforms.Lambda(lambda x: imo.invert(x)),  # Invert
        transforms.Lambda(lambda x: imo.equalize(x)),  # Equalize
        transforms.Lambda(lambda x: imo.solarize(x)),  # Solarize
        transforms.Lambda(lambda x: imo.posterize(
            x, bits=int(np.random.randint(4, 8) + 1))),  # Posterize
        transforms.Lambda(lambda x: ime.Color(
            x).enhance(np.random.uniform())),  # Color
        transforms.Lambda(lambda x: ime.Brightness(
            x).enhance(np.random.uniform())),  # Brightness
        transforms.Lambda(lambda x: ime.Contrast(
            x).enhance(np.random.uniform())),  # Contrast
        transforms.Lambda(lambda x: ime.Sharpness(
            x).enhance(np.random.uniform())),  # Sharpness
        transforms.Compose(  # Set black
        [transforms.ToTensor(), transforms.RandomErasing(1.0), transforms.ToPILImage()])
    ]

    # 1.
    t1 = transforms.RandomChoice(pool)
    t2 = transforms.RandomChoice(pool)
    t3 = transforms.RandomChoice(pool)

    img = t3(t2(t1(img)))

    # https://www.nature.com/articles/s41591-018-0107-6
    # 2. Only elastic def, no shearing as this is part of pool as well as hist changes
    if np.random.uniform() < 0.2:
        img = elastic_transform(img, sigma=10)

    # 3. In pool
    # 4.
    if np.random.uniform() < 0.25:
        img = transforms.functional.to_grayscale(img, num_output_channels=3)

    return img


def PIRLAugmentations(img):
    s = img.size[0] // 3
    crop_size = 64

    t1 = transforms.Compose([transforms.RandomCrop(
        crop_size, pad_if_needed=True), transforms.Resize((s, s)),
        transforms.Lambda(lambda x: ContrastivePredictiveCodingAugmentations(x))])
    perm = [i for i in range(9)]
    random.shuffle(perm)

    img = jigsaw(img, perm, s, trans=t1, normed=False)
    return img

def JigsawAugmentations(img):

    pool = transforms.Compose([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
                               transforms.RandomGrayscale(p=0.3)])

    img = pool(img)
    return img
