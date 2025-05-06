# 可用于实现Python库的本地安装：
# cd PyMRI
# pip install -e .

from setuptools import setup, find_packages

setup(
    name="PyMRI",
    version="0.1.0",
    author="Yuyang Hua",
    description="Python package for post-processing of MRI-driven turbulence simulation",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.20.0",
        "matplotlib>=3.2.0"
    ],
    python_requires=">=3.6"
) 