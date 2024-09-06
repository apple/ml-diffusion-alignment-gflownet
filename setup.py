# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.

from setuptools import setup, find_packages

setup(
    name="diffusion-alignment-pytorch",
    version="0.0.1",
    python_requires=">=3.8",
    install_requires=[
        "ml-collections",
        "absl-py",
        "diffusers[torch]>=0.29.0", # 0.29.0 supports SD3
        "accelerate",
        "torchvision",
        "inflect==6.0.4",
        "pydantic==1.10.13",

        "wandb",
        "ipdb",
        "line_profiler",
        "timm",
        "termcolor",
        "openai-clip",
        "image-reward",
        "ipykernel",
        "clint",
        "torchmetrics[image]>=1.4.0", # using [image] to install torch-fidelity
        "peft>=0.6.0",
        "transformers>=4.41.2"
        "einops",
        "torchdiffeq",
    ],
)
