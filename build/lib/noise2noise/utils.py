#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as tvF

import os
import numpy as np
from math import log10
from datetime import datetime
import OpenEXR
from PIL import Image
import Imath

from matplotlib import rcParams
rcParams['font.family'] = 'serif'
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


def clear_line():
    """Clears line from any characters."""

    print('\r{}'.format(' ' * 80), end='\r')


def progress_bar(batch_idx, num_batches, report_interval, train_loss):
    """Neat progress bar to track training."""

    dec = int(np.ceil(np.log10(num_batches)))
    bar_size = 21 + dec
    progress = (batch_idx % report_interval) / report_interval
    fill = int(progress * bar_size) + 1
    print('\rBatch {:>{dec}d} [{}{}] Train loss: {:>1.5f}'.format(batch_idx + 1, '=' * fill + '>', ' ' * (bar_size - fill), train_loss, dec=str(dec)), end='')


def time_elapsed_since(start):
    """Computes elapsed time since start."""

    timedelta = datetime.now() - start
    string = str(timedelta)[:-7]
    ms = int(timedelta.total_seconds() * 1000)

    return string, ms


def show_on_epoch_end(epoch_time, valid_time, valid_loss, valid_psnr):
    """Formats validation error stats."""

    clear_line()
    print('Train time: {} | Valid time: {} | Valid loss: {:>1.5f} | Avg PSNR: {:.2f} dB'.format(epoch_time, valid_time, valid_loss, valid_psnr))


def show_on_report(batch_idx, num_batches, loss, elapsed):
    """Formats training stats."""

    clear_line()
    dec = int(np.ceil(np.log10(num_batches)))
    print('Batch {:>{dec}d} / {:d} | Avg loss: {:>1.5f} | Avg train time / batch: {:d} ms'.format(batch_idx + 1, num_batches, loss, int(elapsed), dec=dec))


def plot_per_epoch(ckpt_dir, title, measurements, y_label):
    """Plots stats (train/valid loss, avg PSNR, etc.)."""

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(range(1, len(measurements) + 1), measurements)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlabel('Epoch')
    ax.set_ylabel(y_label)
    ax.set_title(title)
    plt.tight_layout()

    fname = '{}.png'.format(title.replace(' ', '-').lower())
    plot_fname = os.path.join(ckpt_dir, fname)
    plt.savefig(plot_fname, dpi=200)
    plt.close()


def load_hdr_as_tensor(img_path):
    """Converts OpenEXR image to torch float tensor."""

    # Read OpenEXR file
    if not OpenEXR.isOpenExrFile(img_path):
        raise ValueError(f'Image {img_path} is not a valid OpenEXR file')
    src = OpenEXR.InputFile(img_path)
    pixel_type = Imath.PixelType(Imath.PixelType.FLOAT)
    dw = src.header()['dataWindow']
    size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)
    
    # Read into tensor
    tensor = torch.zeros((3, size[1], size[0]))
    for i, c in enumerate('RGB'):
        rgb32f = np.fromstring(src.channel(c, pixel_type), dtype=np.float32)
        tensor[i, :, :] = torch.from_numpy(rgb32f.reshape(size[1], size[0]))
        
    return tensor


def reinhard_tonemap(tensor):
    """Reinhard et al. (2002) tone mapping."""

    tensor[tensor < 0] = 0
    return torch.pow(tensor / (1 + tensor), 1 / 2.2)

def psnr(input, target):
    """Computes peak signal-to-noise ratio."""
    input_np = input.detach().cpu().numpy()
    target_np = target.detach().cpu().numpy()
    mse = np.mean((input_np - target_np) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * np.log10(PIXEL_MAX / np.sqrt(mse))

def create_montage(img_name, noise_type, save_path, source_t, denoised_t, clean_t=None, show=0):
    """Creates montage for easy comparison."""
    if clean_t is not None:
        fig, ax = plt.subplots(2, 3, figsize=(9, 7))
    else:
        fig, ax = plt.subplots(2, 2, figsize=(9, 7))

    # Detach tensors and bring them to CPU
    source_t = source_t.detach().cpu().narrow(0, 0, 3)
    denoised_t = denoised_t.detach().cpu()
    if clean_t is not None:
        clean_t = clean_t.detach().cpu()
    
    # Convert tensors to PIL images
    source = tvF.to_pil_image(source_t)
    denoised = tvF.to_pil_image(torch.clamp(denoised_t, 0, 1))
    
    # Build image montage
    titles = ['Input', 'Denoised']
    images = [source, denoised]
    psnr_vals = []

    if clean_t is not None:
        clean = tvF.to_pil_image(clean_t)
        titles.append('Ground truth')
        images.append(clean)
        psnr_vals.append(psnr(clean_t, denoised_t))
        titles[1] += f': {psnr_vals[-1]:.2f} dB'
    
    plot_data_list = []
    min_y = []
    max_y = []

    for j, (title, img) in enumerate(zip(titles, images)):
        ax[0][j].imshow(img)
        ax[0][j].axhline(y=img.size[1] // 2, color='red', lw=2, ls='-')
        ax[0][j].set_title(title)
        ax[0][j].axis('off')

        img_array = np.array(img)
        plot_data = img_array[img.size[1] // 2]
        plot_data_list.append(plot_data)
        min_y.append(plot_data.min())
        max_y.append(plot_data.max())

    global_min_y = min(min_y)
    global_max_y = max(max_y)

    for j in range(len(plot_data_list)):
        ax[1][j].plot(plot_data_list[j], lw=1, ls='-', color='red')
        ax[1][j].set_ylim(global_min_y, global_max_y)
        ax[1][j].axis('on')

    if show > 0:
        plt.show()

    fname = os.path.splitext(img_name)[0]
    source.save(os.path.join(save_path, f'{fname}-{noise_type}-noisy.png'))
    denoised.save(os.path.join(save_path, f'{fname}-{noise_type}-denoised.png'))
    if clean_t is not None:
        clean.save(os.path.join(save_path, f'{fname}-{noise_type}-clean.png'))
    fig.savefig(os.path.join(save_path, f'{fname}-{noise_type}-montage.png'), bbox_inches='tight')




class AvgMeter(object):
    """Computes and stores the average and current value.
    Useful for tracking averages such as elapsed times, minibatch losses, etc.
    """

    def __init__(self):
        self.reset()


    def reset(self):
        self.val = 0
        self.avg = 0.
        self.sum = 0
        self.count = 0


    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count