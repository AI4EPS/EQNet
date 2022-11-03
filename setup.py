from setuptools import setup

setup(
    name="EQNet",
    version="0.1.1",
    long_description="EQNet: Neural Network Models for Earthquakes",
    long_description_content_type="text/markdown",
    packages=["eqnet"],
    install_requires=["torch", "torchvision", "h5py", "matplotlib", "pandas"],
)
