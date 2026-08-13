#!/usr/bin/env python3
"""
SiLens SDK Setup Script.

This setup.py is provided for compatibility with older tools.
For modern installations, pyproject.toml is preferred.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text() if readme_path.exists() else ""

# Read version from package
version = "0.1.0"
try:
    with open("silens/__init__.py") as f:
        for line in f:
            if line.startswith("__version__"):
                version = line.split("=")[1].strip().strip('"\'')
                break
except FileNotFoundError:
    pass

setup(
    name="silens",
    version=version,
    author="SiLens Team",
    author_email="team@silens.ai",
    description="Python SDK for SiLens Vision-Language AI Accelerator",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/silens/silens-sdk",
    project_urls={
        "Documentation": "https://docs.silens.ai",
        "Bug Tracker": "https://github.com/silens/silens-sdk/issues",
    },
    packages=find_packages(exclude=["tests*", "examples*"]),
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.24.0",
        "pillow>=10.0.0",
    ],
    extras_require={
        "full": [
            "transformers>=4.40.0",
            "torch>=2.0.0",
            "huggingface_hub>=0.20.0",
        ],
        "usb": [
            "pyusb>=1.2.0",
        ],
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.5.0",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: System :: Hardware :: Hardware Drivers",
    ],
    keywords="ai accelerator vision language model hardware",
    entry_points={
        "console_scripts": [
            "silens-info=silens.cli:info_cmd",
            "silens-benchmark=silens.cli:benchmark_cmd",
        ],
    },
)
