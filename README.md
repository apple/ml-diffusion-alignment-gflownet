# Diffusion Alignment GFlowNet

This is the PyTorch implementation for our paper [Improving GFlowNets for Text-to-Image Diffusion Alignment
](https://arxiv.org/abs/2406.00633).

This work proposes methods to align text-to-image diffusion models with given reward functions 
through the algorithmic framework of GFlowNet. 
We provide code for DAG (diffusion alignment with gflownet) 
and DAG-KL (DAG with KL divergence based gradient). For more details, we refer to our paper.

## Installation

```bash
pip install -e .
```

## Usage

Diffusion alignment training with GFlowNet on Stable Diffusion:
```bash
torchrun --standalone --nproc_per_node=8 scripts/train_gfn.py
```
To use DAG-KL, set `config['train']['klpf]` in `config/sd.yaml` to a positive coefficient.


## Important Hyperparameters

A detailed explanation of all the hyperparameters can be found in `config/sd.yaml`. 

### prompt_fn and reward_fn
At a high level, the problem of finetuning a diffusion model is defined by 2 things: 
a set of prompts to generate images, and a reward function to evaluate those images. 
The prompts are defined by a `prompt_fn` which takes no arguments and 
generates a random prompt each time it is called. 
The reward function is defined by a `reward_fn` which takes in a batch of images and returns 
a batch of rewards for those images. All of the prompt and reward functions currently implemented can be
found in `alignment/prompts.py` and `alignment/rewards.py`, respectively.

## Acknowledgements

We thank the authors of the [ddpo-pytorch](https://github.com/kvablack/ddpo-pytorch) repository for open sourcing their code, 
which part of our code is based on.


# Citation
If you find this code useful, please consider citing our paper:
```
@article{diffusion_alignment_gfn,
  title={Improving GFlowNets for Text-to-Image Diffusion Alignment},
  author={Dinghuai Zhang and Yizhe Zhang and Jiatao Gu and Ruixiang Zhang and Josh Susskind and Navdeep Jaitly and Shuangfei Zhai},
  journal={Arxiv},
  year={2024},
  url={https://arxiv.org/abs/2406.00633}, 
}
```