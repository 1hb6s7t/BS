from setuptools import setup, find_packages

setup(
    name="srgan",
    version="1.0",
    packages=find_packages(),
    install_requires=[
        "mindspore",
        "numpy",
        "scikit-image",
        "pillow"
    ],
) 