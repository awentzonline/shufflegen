import os

from setuptools import setup, find_packages


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
    name="shufflegen",
    version="0.0.1",
    author="Adam Wentz",
    author_email="adam@adamwentz.com",
    description="Visual embeddings for product images",
    long_description=read("README.md"),
    license="MIT",
    url="https://github.com/awentzonline/shufflegen",
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pytorch-lightning',
        'pytorch-lightning-bolts',
        'torch',
        'torchvision',
        'tqdm',
    ]
)
