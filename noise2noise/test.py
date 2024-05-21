import torch
import torch.nn as nn
from noise2noise import Noise2Noise
from argparse import ArgumentParser
from torchvision import transforms
from PIL import Image
import os
from utils import create_montage  # Import the create_montage function from utils

def parse_args():
    """Command-line argument parser for testing."""

    parser = ArgumentParser(description='PyTorch implementation of Noise2Noise from Lehtinen et al. (2018)')

    # Data parameters
    parser.add_argument('--test-image', help='path to the test image', required=True)
    parser.add_argument('--load-ckpt', help='load model checkpoint', required=True)
    parser.add_argument('--show-output', help='pop up window to display outputs', default=0, type=int)
    parser.add_argument('--cuda', help='use cuda', action='store_true')

    # Corruption parameters
    parser.add_argument('--noise-type', help='noise type', choices=['gaussian', 'poisson', 'text', 'mc'], default='gaussian', type=str)
    parser.add_argument('--noise-param', help='noise parameter (e.g. sigma for gaussian)', default=50, type=float)
    parser.add_argument('--seed', help='fix random seed', type=int)
    parser.add_argument('--crop-size', help='image crop size', default=256, type=int)
    parser.add_argument('--no-crop', help='do not crop images during testing', action='store_true')
    parser.add_argument('--add-noise', help='add noise to images during testing', action='store_true')

    return parser.parse_args()

def load_test_image(image_path, crop_size, no_crop):
    """Loads and preprocesses the test image."""
    img = Image.open(image_path).convert('RGB')
    
    if not no_crop and crop_size > 0:
        # Center crop if no_crop is not set
        transform = transforms.Compose([
            transforms.CenterCrop(crop_size),
            transforms.ToTensor()
        ])
    else:
        # Just convert to tensor
        transform = transforms.Compose([
            transforms.ToTensor()
        ])
    
    img_tensor = transform(img).unsqueeze(0)  # Add batch dimension
    return img_tensor

if __name__ == '__main__':
    """Tests Noise2Noise."""

    # Parse test parameters
    params = parse_args()

    # Load the test image
    test_image = load_test_image(params.test_image, params.crop_size, params.no_crop)

    # Initialize model and test
    n2n = Noise2Noise(params, trainable=False)
    n2n.load_model(params.load_ckpt)

    # Move the test image to GPU if available
    if params.cuda and torch.cuda.is_available():
        test_image = test_image.cuda()

    # Denoise the image
    n2n.model.eval()
    with torch.no_grad():
        denoised_image = n2n.model(test_image).cpu().squeeze()

    # Create the montage
    create_montage(
        img_name=os.path.basename(params.test_image),
        noise_type=params.noise_type,
        save_path=os.path.dirname(params.test_image),
        source_t=test_image.squeeze(),
        denoised_t=denoised_image,
        show=params.show_output
    )