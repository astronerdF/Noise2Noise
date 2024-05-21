from setuptools import setup, find_packages


setup(
    name='nosie2noise',
    version='1.0',
    packages=find_packages(),
    install_requires=[
        'torch',
        'torchvision',
        'numpy',
        'Pillow',
        'matplotlib',
        'OpenEXR',
    ]
)