import torch
import torch.nn as nn
from .datasets import load_dataset
from .noise2noise import Noise2Noise
from argparse import ArgumentParser

def parse_args():
    """Command-line argument parser for training."""

    parser = ArgumentParser(description='PyTorch implementation of Noise2Noise from Lehtinen et al. (2018)')

    # Data parameters
    parser.add_argument('--train-source-dir', help='training source images path', default='data/train/source')
    parser.add_argument('--train-target-dir', help='training target images path', default='data/train/target')
    parser.add_argument('--valid-source-dir', help='validation source images path', default='data/valid/source')
    parser.add_argument('--valid-target-dir', help='validation target images path', default='data/valid/target')
    parser.add_argument('--ckpt-save-path', help='checkpoint save path', default='ckpts')
    parser.add_argument('--ckpt-overwrite', help='overwrite model checkpoint on save', action='store_true')
    parser.add_argument('--report-interval', help='batch report interval', default=25, type=int)
    parser.add_argument('--train-size', help='size of train dataset', type=int)
    parser.add_argument('--valid-size', help='size of valid dataset', type=int)

    # Training hyperparameters
    parser.add_argument('--learning-rate', help='learning rate', default=0.001, type=float)
    parser.add_argument('--adam', help='adam parameters', nargs='+', default=[0.9, 0.99, 1e-8], type=list)
    parser.add_argument('--batch-size', help='minibatch size', default=4, type=int)
    parser.add_argument('--nb-epochs', help='number of epochs', default=30, type=int)
    parser.add_argument('--loss', help='loss function', choices=['l1', 'l2', 'hdr'], default='l2', type=str)
    parser.add_argument('--cuda', help='use cuda', action='store_true')
    parser.add_argument('--plot-stats', help='plot stats after every epoch', action='store_true')

    # Corruption parameters
    parser.add_argument('--noise-type', help='noise type', choices=['gaussian', 'poisson', 'text', 'mc'], default='gaussian', type=str)
    parser.add_argument('--noise-param', help='noise parameter (e.g. std for gaussian)', default=50, type=float)
    parser.add_argument('--seed', help='fix random seed', type=int)
    parser.add_argument('--crop-size', help='random crop size', default=128, type=int)
    parser.add_argument('--clean-targets', help='use clean targets for training', action='store_true')
    parser.add_argument('--no-crop', help='do not crop images during testing', action='store_true')
    parser.add_argument('--add-noise', help='add noise to images during training', action='store_true')

    return parser.parse_args()

if __name__ == '__main__':
    """Trains Noise2Noise."""

    # Parse training parameters
    params = parse_args()

    # Train/valid datasets
    train_loader = load_dataset(
        source_dir=params.train_source_dir,
        target_dir=params.train_target_dir,
        redux=params.train_size,
        params=params,
        shuffled=True,
        add_noise=params.add_noise
    )
    valid_loader = load_dataset(
        source_dir=params.valid_source_dir,
        target_dir=params.valid_target_dir,
        redux=params.valid_size,
        params=params,
        shuffled=False,
        add_noise=params.add_noise
    )

    # Initialize model and train
    n2n = Noise2Noise(params, trainable=True)
    n2n.train(train_loader, valid_loader)