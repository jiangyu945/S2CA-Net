import os
import torch
from torch.utils.data import Dataset
import random
import numpy as np
from torchvision.transforms import transforms
import pickle
from scipy import ndimage
from glob import glob

import nibabel as nib
import cv2


def pkload(fname):
    with open(fname, 'rb') as f:
        return pickle.load(f)


class MaxMinNormalization(object):
    def __call__(self, sample):
        image = sample['image']
        label = sample['label']
        Max = np.max(image)
        Min = np.min(image)
        image = (image - Min) / (Max - Min)

        return {'image': image, 'label': label}


class Random_Flip(object):
    def __call__(self, sample):
        image = sample['image']
        label = sample['label']
        if random.random() < 0.5:
            image = np.flip(image, 0)
            label = np.flip(label, 0)
        if random.random() < 0.5:
            image = np.flip(image, 1)
            label = np.flip(label, 1)
        if random.random() < 0.5:
            image = np.flip(image, 2)
            label = np.flip(label, 2)

        return {'image': image, 'label': label}


class Random_Crop(object):
    def __call__(self, sample):
        crop_size = 128  # 128 - cut to (128, 128, 128) 
        image = sample['image']
        label = sample['label']
        H = random.randint(0, 240 - crop_size)
        W = random.randint(0, 240 - crop_size)
        D = random.randint(0, 160 - crop_size)

        image = image[H: H + crop_size, W: W + crop_size, D: D + crop_size, ...]
        label = label[..., H: H + crop_size, W: W + crop_size, D: D + crop_size]

        return {'image': image, 'label': label}


class Random_intencity_shift(object):
    def __call__(self, sample, factor=0.1):
        image = sample['image']
        label = sample['label']

        scale_factor = np.random.uniform(1.0-factor, 1.0+factor, size=[1, image.shape[1], 1, image.shape[-1]])
        shift_factor = np.random.uniform(-factor, factor, size=[1, image.shape[1], 1, image.shape[-1]])

        image = image*scale_factor+shift_factor

        return {'image': image, 'label': label}


class Random_rotate(object):
    def __call__(self, sample):
        image = sample['image']
        label = sample['label']

        angle = round(np.random.uniform(-10, 10), 2)
        image = ndimage.rotate(image, angle, axes=(0, 1), reshape=False)
        label = ndimage.rotate(label, angle, axes=(0, 1), reshape=False)

        return {'image': image, 'label': label}


class Pad(object):
    def __call__(self, sample):
        image = sample['image']
        label = sample['label']

        image = np.pad(image, ((0, 0), (0, 0), (0, 5), (0, 0)), mode='constant')
        label = np.pad(label, ((0, 0), (0, 0), (0, 5)), mode='constant')
        return {'image': image, 'label': label}
    #(240,240,155)>(240,240,160)


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, sample):
        image = sample['image']
        image = np.ascontiguousarray(image.transpose(3, 0, 1, 2))  # (B, H, W, D, C) -> (B, C, H, W, D)
        label = sample['label']
        label = np.ascontiguousarray(label)

        image = torch.from_numpy(image).float()
        label = torch.from_numpy(label).long()

        return {'image': image, 'label': label}


def transform(sample):
    trans = transforms.Compose([
        Pad(),
        # Random_rotate(),  # time-consuming
        Random_Crop(),
        Random_Flip(),
        Random_intencity_shift(),
        ToTensor()
    ])

    return trans(sample)


def transform_valid(sample):
    trans = transforms.Compose([
        # MaxMinNormalization(),
        ToTensor()
    ])

    return trans(sample)


class BraTS(Dataset):
    def __init__(self, img_ids, mode='train'):
        self.mode = mode
        self.paths = img_ids
        self.names = []
        for id_path in img_ids:
            name = id_path.split('/')[-1]
            self.names.append(name)

    def __getitem__(self, item):
        path = self.paths[item]
        if self.mode == 'train':
            image, label = pkload(path)
            sample = {'image': image, 'label': label}
            sample = transform(sample)
            return sample['image'], sample['label']
        elif self.mode == 'valid':
            image, label = pkload(path)
            sample = {'image': image, 'label': label}
            sample = transform_valid(sample)
            return sample['image'], sample['label']    # image: (4, 240, 240, 155)  label: (240, 240, 155)
        else:
            image = pkload(path)
            image = np.pad(image, ((0, 0), (0, 0), (0, 5), (0, 0)), mode='constant')  # (240, 240, 155, 4) -> (240, 240, 160, 4)
            image = np.ascontiguousarray(image.transpose(3, 0, 1, 2))  # (B, H, W, D, C) -> (B, C, H, W, D)
            image = torch.from_numpy(image).float()  # convert to tensor
            return image

    def __len__(self):
        return len(self.paths)

    def collate(self, batch):
        return [torch.cat(v) for v in zip(*batch)]


if __name__ == '__main__':

    train_root = '/home/ff/jy/datasets/Preprocess_BraTS/2018/3D/train'

    img_ids = glob(os.path.join(train_root, '*.pkl'))
    # train_img_ids, val_img_ids = train_test_split(img_ids, test_size=0.2, random_state=99)
    train_set = BraTS(img_ids, 'train')
    print('Number of samples: ', len(train_set))

    train_loader = torch.utils.data.DataLoader(dataset=train_set, shuffle=True, batch_size=16,
                                               drop_last=True, num_workers=4, pin_memory=True)



    for i, (img, mask) in enumerate(train_loader):
        print('img.shape: ', img.shape)  # img: [B, C, D, H, W]  [16, 4, 128, 128, 128]
        print('mask.shape: ', mask.shape)  # mask: [B, D, H, W]  [16,128, 128, 128]

        # nib方式写入
        nib.save(nib.Nifti1Image(img[0,0,:,:,:].numpy(), None), 'img/1.nii.gz')
        nib.save(nib.Nifti1Image(mask[0, :, :, :].numpy(), None), 'img/1_seg.nii.gz')

        break


