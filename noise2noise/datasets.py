#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as tvF
from torch.utils.data import Dataset, DataLoader

from .utils import load_hdr_as_tensor

import os
from sys import platform
import numpy as np
import random
from string import ascii_letters
from PIL import Image, ImageFont, ImageDraw
import OpenEXR

from matplotlib import rcParams
rcParams['font.family'] = 'serif'
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt


def load_dataset(source_dir, target_dir, redux, params, shuffled=False, single=False, no_crop=False, add_noise=True):
    """Loads dataset and returns corresponding data loader."""

    # Create Torch dataset
    noise = (params.noise_type, params.noise_param)

    # Use clean_targets if it exists, otherwise default to False
    clean_targets = getattr(params, 'clean_targets', False)

    dataset = NoisyDataset(
        source_dir=source_dir,
        target_dir=target_dir,
        redux=redux,
        crop_size=params.crop_size,
        clean_targets=clean_targets,
        noise_dist=noise,
        seed=params.seed,
        no_crop=no_crop,
        add_noise=add_noise
    )

    # Use batch size of 1, if requested (e.g. test set)
    if single:
        return DataLoader(dataset, batch_size=1, shuffle=shuffled)
    else:
        return DataLoader(dataset, batch_size=params.batch_size, shuffle=shuffled)






class AbstractDataset(Dataset):
    """Abstract dataset class for Noise2Noise."""

    def __init__(self, root_dir, redux=0, crop_size=128, clean_targets=False):
        """Initializes abstract dataset."""

        super(AbstractDataset, self).__init__()

        self.imgs = []
        self.root_dir = root_dir
        self.redux = redux
        self.crop_size = crop_size
        self.clean_targets = clean_targets

    def _random_crop(self, img_list):
        """Performs random square crop of fixed size.
        Works with list so that all items get the same cropped window (e.g. for buffers).
        """

        w, h = img_list[0].size
        assert w >= self.crop_size and h >= self.crop_size, \
            f'Error: Crop size: {self.crop_size}, Image size: ({w}, {h})'
        cropped_imgs = []
        i = np.random.randint(0, h - self.crop_size + 1)
        j = np.random.randint(0, w - self.crop_size + 1)

        for img in img_list:
            # Resize if dimensions are too small
            if min(w, h) < self.crop_size:
                img = tvF.resize(img, (self.crop_size, self.crop_size))

            # Random crop
            cropped_imgs.append(tvF.crop(img, i, j, self.crop_size, self.crop_size))

        return cropped_imgs


    def __getitem__(self, index):
        """Retrieves image from data folder."""

        raise NotImplementedError('Abstract method not implemented!')


    def __len__(self):
        """Returns length of dataset."""

        return len(self.imgs)


class NoisyDataset(AbstractDataset):
    """Class for loading pairs of noisy images from two directories."""

    def __init__(self, source_dir, target_dir, redux, crop_size, clean_targets=False,
                 noise_dist=('gaussian', 50.), seed=None, no_crop=False, add_noise=True):
        """Initializes noisy image dataset."""

        super(NoisyDataset, self).__init__(source_dir, redux, crop_size, clean_targets)

        self.source_dir = source_dir
        self.target_dir = target_dir
        self.source_imgs = sorted(os.listdir(source_dir))
        self.target_imgs = sorted(os.listdir(target_dir))

        if redux:
            self.source_imgs = self.source_imgs[:redux]
            self.target_imgs = self.target_imgs[:redux]

        self.noise_type = noise_dist[0]
        self.noise_param = noise_dist[1]
        self.seed = seed
        self.no_crop = no_crop
        self.add_noise = add_noise
        if self.seed:
            np.random.seed(self.seed)

    def _add_noise(self, img):
        """Adds Gaussian or Poisson noise to image."""

        w, h = img.size
        c = len(img.getbands())

        # Poisson distribution
        # It is unclear how the paper handles this. Poisson noise is not additive,
        # it is data dependent, meaning that adding sampled valued from a Poisson
        # will change the image intensity...
        if self.noise_type == 'poisson':
            noise = np.random.poisson(img)
            noise_img = img + noise
            noise_img = 255 * (noise_img / np.amax(noise_img))

        # Normal distribution (default)
        else:
            if self.seed:
                std = self.noise_param
            else:
                std = np.random.uniform(0, self.noise_param)
            noise = np.random.normal(0, std, (h, w, c))

            # Add noise and clip
            noise_img = np.array(img) + noise

        noise_img = np.clip(noise_img, 0, 255).astype(np.uint8)
        return Image.fromarray(noise_img)


    def _add_text_overlay(self, img):
        """Adds text overlay to images."""

        assert self.noise_param < 1, 'Text parameter is an occupancy probability'

        w, h = img.size
        c = len(img.getbands())

        # Choose font and get ready to draw
        if platform == 'linux':
            serif = '/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf'
        else:
            serif = 'Times New Roman.ttf'
        text_img = img.copy()
        text_draw = ImageDraw.Draw(text_img)

        # Text binary mask to compute occupancy efficiently
        w, h = img.size
        mask_img = Image.new('1', (w, h))
        mask_draw = ImageDraw.Draw(mask_img)

        # Random occupancy in range [0, p]
        if self.seed:
            random.seed(self.seed)
            max_occupancy = self.noise_param
        else:
            max_occupancy = np.random.uniform(0, self.noise_param)
        def get_occupancy(x):
            y = np.array(x, dtype=np.uint8)
            return np.sum(y) / y.size

        # Add text overlay by choosing random text, length, color and position
        while 1:
            font = ImageFont.truetype(serif, np.random.randint(16, 21))
            length = np.random.randint(10, 25)
            chars = ''.join(random.choice(ascii_letters) for i in range(length))
            color = tuple(np.random.randint(0, 255, c))
            pos = (np.random.randint(0, w), np.random.randint(0, h))
            text_draw.text(pos, chars, color, font=font)

            # Update mask and check occupancy
            mask_draw.text(pos, chars, 1, font=font)
            if get_occupancy(mask_img) > max_occupancy:
                break

        return text_img

# Adds noise to the Image
    def _corrupt(self, img):
       #Corrupts images (Gaussian, Poisson, or text overlay).

        if self.noise_type in ['gaussian', 'poisson']:
            return self._add_noise(img)
        elif self.noise_type == 'text':
            return self._add_text_overlay(img)
        else:
            raise ValueError('Invalid noise type: {}'.format(self.noise_type)) 

# Does not add noise to the image
#    def _corrupt(self, img):
#        """Returns the image unchanged, skipping any corruption."""
#        return img

        


    def __getitem__(self, index):
        """Retrieves image pair from folders."""

        source_img_path = os.path.join(self.source_dir, self.source_imgs[index])
        target_img_path = os.path.join(self.target_dir, self.target_imgs[index])

        source_img = Image.open(source_img_path).convert('RGB')
        target_img = Image.open(target_img_path).convert('RGB')

        if not self.no_crop and self.crop_size != 0:
            source_img, target_img = self._random_crop([source_img, target_img])

        source = tvF.to_tensor(source_img)
        target = tvF.to_tensor(target_img)

        return source, target

    def __len__(self):
        """Returns length of dataset."""

        return len(self.source_imgs)


class MonteCarloDataset(AbstractDataset):
    """Class for dealing with Monte Carlo rendered images."""

    def __init__(self, root_dir, redux, crop_size,
        hdr_buffers=False, hdr_targets=True, clean_targets=False):
        """Initializes Monte Carlo image dataset."""

        super(MonteCarloDataset, self).__init__(root_dir, redux, crop_size, clean_targets)

        # Rendered images directories
        self.root_dir = root_dir
        self.imgs = os.listdir(os.path.join(root_dir, 'render'))
        self.albedos = os.listdir(os.path.join(root_dir, 'albedo'))
        self.normals = os.listdir(os.path.join(root_dir, 'normal'))

        if redux:
            self.imgs = self.imgs[:redux]
            self.albedos = self.albedos[:redux]
            self.normals = self.normals[:redux]

        # Read reference image (converged target)
        ref_path = os.path.join(root_dir, 'reference.png')
        self.reference = Image.open(ref_path).convert('RGB')

        # High dynamic range images
        self.hdr_buffers = hdr_buffers
        self.hdr_targets = hdr_targets


    def __getitem__(self, index):
        """Retrieves image from folder and corrupts it."""

        # Use converged image, if requested
        if self.clean_targets:
            target = self.reference
        else:
            target_fname = self.imgs[index].replace('render', 'target')
            file_ext = '.exr' if self.hdr_targets else '.png'
            target_fname = os.path.splitext(target_fname)[0] + file_ext
            target_path = os.path.join(self.root_dir, 'target', target_fname)
            if self.hdr_targets:
                target = tvF.to_pil_image(load_hdr_as_tensor(target_path))
            else:
                target = Image.open(target_path).convert('RGB')

        # Get buffers
        render_path = os.path.join(self.root_dir, 'render', self.imgs[index])
        albedo_path = os.path.join(self.root_dir, 'albedo', self.albedos[index])
        normal_path =  os.path.join(self.root_dir, 'normal', self.normals[index])

        if self.hdr_buffers:
            render = tvF.to_pil_image(load_hdr_as_tensor(render_path))
            albedo = tvF.to_pil_image(load_hdr_as_tensor(albedo_path))
            normal = tvF.to_pil_image(load_hdr_as_tensor(normal_path))
        else:
            render = Image.open(render_path).convert('RGB')
            albedo = Image.open(albedo_path).convert('RGB')
            normal = Image.open(normal_path).convert('RGB')

        # Crop
        if self.crop_size != 0:
            buffers = [render, albedo, normal, target]
            buffers = [tvF.to_tensor(b) for b in self._random_crop(buffers)]

        # Stack buffers to create input volume
        source = torch.cat(buffers[:3], dim=0)
        target = buffers[3]

        return source, target