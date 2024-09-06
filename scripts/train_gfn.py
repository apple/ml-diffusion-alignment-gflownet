# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.

import os, sys
from collections import defaultdict
import contextlib
import datetime
import time
from concurrent import futures
import wandb
from functools import partial
import tempfile
from PIL import Image
import tqdm
tqdm = partial(tqdm.tqdm, dynamic_ncols=True)
import logging
import yaml
from termcolor import colored
import copy
import math
import pickle, gzip

import diffusers
from diffusers import AutoencoderKL, DDPMScheduler, DDIMScheduler, StableDiffusionPipeline, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from diffusers.training_utils import cast_training_params
from diffusers.utils import check_min_version, convert_state_dict_to_diffusers, is_wandb_available
from diffusers.utils.torch_utils import is_compiled_module

import datasets
from packaging import version
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict
import transformers
from transformers import CLIPTextModel, CLIPTokenizer

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from scripts.distributed import init_distributed_singlenode, set_seed, load_distributed, setup_for_distributed

import alignment.prompts
import alignment.rewards
from alignment.diffusers_patch.pipeline_with_logprob import pipeline_with_logprob
from alignment.diffusers_patch.ddim_with_logprob import ddim_step_with_logprob, pred_orig_latent
from alignment.flow import ConditionalFlow


def unwrap_model(model):
    model = model.module if isinstance(model, DDP) else model
    model = model._orig_mod if is_compiled_module(model) else model
    return model

def main():
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger = logging.getLogger(__name__)

    config = yaml.safe_load(open("config/sd.yaml"))['parameters']

    local_rank, global_rank, world_size = init_distributed_singlenode(timeout=36000)
    num_processes = world_size
    is_local_main_process = local_rank == 0
    setup_for_distributed(is_local_main_process)

    config['gpu_type'] = torch.cuda.get_device_name() \
                            if torch.cuda.is_available() else "CPU"
    logger.info(f"GPU type: {config['gpu_type']}")

    output_dir = os.path.join("./output")
    os.makedirs(output_dir, exist_ok=True)
    if config['wandb']:
        wandb.init(project="gflownet-alignment SD", config=config,
           save_code=True, mode="online" if is_local_main_process else "disabled")

    logger.info(f"\n{config}")
    set_seed(config['seed'])

    # For mixed precision training we cast all non-trainable weights (vae, non-lora text_encoder and non-lora unet) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if config['mixed_precision'] == "fp16":
        weight_dtype = torch.float16
    elif config['mixed_precision'] == "bf16":
        weight_dtype = torch.bfloat16
    device = torch.device(local_rank)

    pipeline = StableDiffusionPipeline.from_pretrained(
        config['pretrained']['model'], revision=config['pretrained']['revision'], torch_dtype=weight_dtype,
    )
    scheduler_config = {}
    scheduler_config.update(pipeline.scheduler.config)
    pipeline.scheduler = DDIMScheduler.from_config(scheduler_config)
    pipeline.vae.requires_grad_(False)
    pipeline.text_encoder.requires_grad_(False)
    pipeline.vae.to(device, dtype=weight_dtype)
    pipeline.text_encoder.to(device, dtype=weight_dtype)

    pipeline.safety_checker = None
    pipeline.set_progress_bar_config(
        position=1,
        disable=not is_local_main_process,
        leave=False,
        desc="Timestep",
        dynamic_ncols=True,
    )

    unet = pipeline.unet
    unet.requires_grad_(False)
    for param in unet.parameters():
        param.requires_grad_(False)
    assert config['use_lora']
    unet.to(device, dtype=weight_dtype)
    unet_lora_config = LoraConfig(
        r=config['train']['lora_rank'], lora_alpha=config['train']['lora_rank'],
        init_lora_weights="gaussian", target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    )
    unet.add_adapter(unet_lora_config)
    if config['mixed_precision'] in ["fp16", "bf16"]:
        # only upcast trainable parameters (LoRA) into fp32
        cast_training_params(unet, dtype=torch.float32)
    lora_layers = filter(lambda p: p.requires_grad, unet.parameters())

    scaler = None
    if config['mixed_precision'] in ["fp16", "bf16"]:
        scaler = torch.cuda.amp.GradScaler()

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if config['allow_tf32']:
        torch.backends.cuda.matmul.allow_tf32 = True
        # torch.backends.cudnn.allow_tf32 is True by default
        torch.backends.cudnn.benchmark = True

    if config['train']['use_8bit_adam']:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )
        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    # prepare prompt and reward fn
    prompt_fn = getattr(alignment.prompts, config['prompt_fn'])
    reward_fn = getattr(alignment.rewards, config['reward_fn'])(weight_dtype, device)

    # generate negative prompt embeddings
    neg_prompt_embed = pipeline.text_encoder(
        pipeline.tokenizer(
            [""],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=pipeline.tokenizer.model_max_length, # 77
        ).input_ids.to(device)
    )[0]
    sample_neg_prompt_embeds = neg_prompt_embed.repeat(config['sample']['batch_size'], 1, 1)
    train_neg_prompt_embeds = neg_prompt_embed.repeat(config['train']['batch_size'], 1, 1)

    # for some reason, autocast is necessary for non-lora training but for lora training it isn't necessary and it uses
    # more memory
    def func_autocast():
        return torch.cuda.amp.autocast(dtype=weight_dtype)
    if config['use_lora']:
        # LoRA weights are actually float32, but other part of SD are in bf16/fp16
        autocast = contextlib.nullcontext
    else:
        autocast = func_autocast

    unet.to(device)
    unet = DDP(unet, device_ids=[local_rank])

    #######################################################
    #################### FOR GFN ##########################
    def decode(latents):
        image = pipeline.vae.decode(
            latents / pipeline.vae.config.scaling_factor, return_dict=False
        )[0]
        # image, has_nsfw_concept = pipeline.run_safety_checker(
        #     image, device, prompt_embeds.dtype
        # )
        do_denormalize = [True] * image.shape[0]
        image = pipeline.image_processor.postprocess(image,
                     output_type="pt", do_denormalize=do_denormalize)
        return image

    flow_model = ConditionalFlow(in_channels=4, block_out_channels=(64, 128, 256, 256),
         layers_per_block=1, cross_attention_dim=pipeline.text_encoder.config.hidden_size) # hidden_size=768 is SD's text enconder output size
    flow_model = flow_model.to(device, dtype=torch.float32)
    autocast_flow = func_autocast

    flow_model = DDP(flow_model, device_ids=[local_rank])
    params = [
        {"params": lora_layers, "lr": config['train']['learning_rate']},
        {"params": flow_model.parameters(), "lr": config['train']['learning_rate']}
    ]
    optimizer = optimizer_cls(
        params,
        betas=(config['train']['adam_beta1'], config['train']['adam_beta2']),
        weight_decay=config['train']['adam_weight_decay'],
        eps=config['train']['adam_epsilon'],
    )

    result = defaultdict(dict)
    result["config"] = config
    start_time = time.time()

    #######################################################
    # Start!
    samples_per_epoch = (
        config['sample']['batch_size'] * num_processes
        * config['sample']['num_batches_per_epoch']
    )
    total_train_batch_size = (
        config['train']['batch_size'] * num_processes
        * config['train']['gradient_accumulation_steps']
    )

    logger.info("***** Running training *****")
    logger.info(f"  Num Epochs = {config['num_epochs']}")
    logger.info(f"  Sample batch size per device = {config['sample']['batch_size']}")
    logger.info(f"  Train batch size per device = {config['train']['batch_size']}")
    logger.info(
        f"  Gradient Accumulation steps = {config['train']['gradient_accumulation_steps']}"
    )
    logger.info("")
    logger.info(f"  Total number of samples per epoch = test_bs * num_batch_per_epoch * num_process = {samples_per_epoch}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = train_bs * grad_accumul * num_process = {total_train_batch_size}"
    )
    logger.info(
        f"  Number of gradient updates per inner epoch = samples_per_epoch // total_train_batch_size = {samples_per_epoch // total_train_batch_size}"
    )
    logger.info(f"  Number of inner epochs = {config['train']['num_inner_epochs']}")

    assert config['sample']['batch_size'] >= config['train']['batch_size']
    assert config['sample']['batch_size'] % config['train']['batch_size'] == 0 # not necessary
    assert samples_per_epoch % total_train_batch_size == 0

    first_epoch = 0
    global_step = 0
    for epoch in range(first_epoch, config['num_epochs']):
        if config['train']['anneal'] in ["linear"]:
            ratio = min(1, epoch / (0.5 * config['num_epochs']))
        else:
            ratio = 1.
        reward_exp_ep = config['train']['reward_exp'] * ratio
        def reward_transform(value):
            return value * reward_exp_ep

        num_diffusion_steps = config['sample']['num_steps']
        pipeline.scheduler.set_timesteps(num_diffusion_steps, device=device)  # set_timesteps(): 1000 steps -> 50 steps
        scheduler_dt = pipeline.scheduler.timesteps[0] - pipeline.scheduler.timesteps[1]
        num_train_timesteps = int(num_diffusion_steps * config['train']['timestep_fraction'])
        accumulation_steps = config['train']['gradient_accumulation_steps'] * num_train_timesteps

        #################### SAMPLING ####################
        torch.cuda.empty_cache()
        unet.zero_grad()
        unet.eval()
        flow_model.zero_grad()

        if True:
            with torch.inference_mode(): # similar to torch.no_grad() but also disables autograd.grad()
                samples = []
                prompts = []
                for i in tqdm(
                    range(config['sample']['num_batches_per_epoch']),
                    desc=f"Epoch {epoch}: sampling",
                    disable=not is_local_main_process,
                    position=0,
                ):
                    # generate prompts
                    prompts, prompt_metadata = zip(
                        *[
                            prompt_fn(**config['prompt_fn_kwargs'])
                            for _ in range(config['sample']['batch_size'])
                        ]
                    )

                    # encode prompts
                    prompt_ids = pipeline.tokenizer(
                        prompts,
                        return_tensors="pt",
                        padding="max_length",
                        truncation=True,
                        max_length=pipeline.tokenizer.model_max_length,
                    ).input_ids.to(device)
                    prompt_embeds = pipeline.text_encoder(prompt_ids)[0]

                    # sample
                    with autocast():
                        ret_tuple = pipeline_with_logprob(
                            pipeline,
                            prompt_embeds=prompt_embeds,
                            negative_prompt_embeds=sample_neg_prompt_embeds,
                            num_inference_steps=num_diffusion_steps,
                            guidance_scale=config['sample']['guidance_scale'],
                            eta=config['sample']['eta'],
                            output_type="pt",

                            return_unetoutput=config['train']['unetreg'] > 0.,
                        )
                    if config['train']['unetreg'] > 0:
                        images, _, latents, log_probs, unet_outputs = ret_tuple
                        unet_outputs = torch.stack(unet_outputs, dim=1)  # (batch_size, num_steps, 3, 32, 32)
                    else:
                        images, _, latents, log_probs = ret_tuple

                    latents = torch.stack(latents, dim=1)  # (batch_size, num_steps + 1, 4, 64, 64)
                    log_probs = torch.stack(log_probs, dim=1)  # (batch_size, num_steps, 1)
                    timesteps = pipeline.scheduler.timesteps.repeat(
                        config['sample']['batch_size'], 1
                    )  # (bs, num_steps)  (981, 961, ..., 21, 1) corresponds to "next_latents"

                    rewards = reward_fn(images, prompts, prompt_metadata) # (reward, reward_metadata)

                    samples.append(
                        {
                            "prompts": prompts, # tuple of strings
                            "prompt_metadata": prompt_metadata,

                            "prompt_ids": prompt_ids,
                            "prompt_embeds": prompt_embeds,
                            "timesteps": timesteps,
                            "latents": latents[
                                :, :-1
                            ],  # each entry is the latent before timestep t
                            "next_latents": latents[
                                :, 1:
                            ],  # each entry is the latent after timestep t
                            "log_probs": log_probs,
                            "rewards": rewards,
                        }
                    )
                    if config['train']['unetreg'] > 0:
                        samples[-1]["unet_outputs"] = unet_outputs

                # wait for all rewards to be computed
                for sample in tqdm(
                    samples,
                    desc="Waiting for rewards",
                    disable=not is_local_main_process,
                    position=0,
                ):
                    rewards, reward_metadata = sample["rewards"]
                    sample["rewards"] = torch.as_tensor(rewards, device=device)

            # collate samples into dict where each entry has shape (num_batches_per_epoch * sample.batch_size, ...)
            new_samples = {}
            for k in samples[0].keys():
                if k in ["prompts", "prompt_metadata"]:
                    # list of tuples [('cat', 'dog'), ('cat', 'tiger'), ...] -> list ['cat', 'dog', 'cat', 'tiger', ...]
                    new_samples[k] = [item for s in samples for item in s[k]]
                else:
                    new_samples[k] = torch.cat([s[k] for s in samples])
            samples = new_samples

            # this is a hack to force wandb to log the images as JPEGs instead of PNGs
            with tempfile.TemporaryDirectory() as tmpdir:
                for i, image in enumerate(images):
                    # bf16 cannot be converted to numpy directly
                    pil = Image.fromarray(
                        (image.cpu().float().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
                    )
                    pil = pil.resize((256, 256))
                    pil.save(os.path.join(tmpdir, f"{i}.jpg"))
                if config['wandb'] and is_local_main_process:
                    wandb.log(
                        {
                            "images": [
                                wandb.Image(
                                    os.path.join(tmpdir, f"{i}.jpg"),
                                    caption=f"{prompt} | {reward:.2f}",
                                )
                                for i, (prompt, reward) in enumerate(
                                    zip(prompts, rewards)
                                )
                            ],
                        },
                        step=global_step,
                    )

            rewards = torch.zeros(world_size * len(samples["rewards"]),
                          dtype=samples["rewards"].dtype, device=device)
            dist.all_gather_into_tensor(rewards, samples["rewards"])
            rewards = rewards.cpu().float().numpy()
            result["reward_mean"][global_step] = rewards.mean()
            result["reward_std"][global_step] = rewards.std()

            if is_local_main_process:
                logger.info(f"global_step: {global_step}  rewards: {rewards.mean().item():.3f}")
                if config['wandb']:
                    wandb.log(
                        {
                            "reward_mean": rewards.mean(), # samples["rewards"].mean()
                            "reward_std": rewards.std(),
                        },
                        step=global_step,
                    )

            del samples["prompt_ids"]

            total_batch_size, num_timesteps = samples["timesteps"].shape
            assert (
                total_batch_size
                == config['sample']['batch_size'] * config['sample']['num_batches_per_epoch']
            )
            assert num_timesteps == num_diffusion_steps

        #################### TRAINING ####################
        for inner_epoch in range(config['train']['num_inner_epochs']):
            # shuffle samples along batch dimension
            perm = torch.randperm(total_batch_size, device=device)
            for k, v in samples.items():
                if k in ["prompts", "prompt_metadata"]:
                    samples[k] = [v[i] for i in perm]
                elif k in ["unet_outputs"]:
                    samples[k] = v[perm]
                else:
                    samples[k] = v[perm]

            perms = torch.stack(
                [
                    torch.randperm(num_timesteps, device=device)
                    for _ in range(total_batch_size)
                ]
            ) # (total_batch_size, num_steps)
            # "prompts" & "prompt_metadata" are constant along time dimension
            key_ls = ["timesteps", "latents", "next_latents", "log_probs"]
            for key in key_ls:
                samples[key] = samples[key][torch.arange(total_batch_size, device=device)[:, None], perms]
            if config['train']['unetreg'] > 0:
                samples["unet_outputs"] = \
                    samples["unet_outputs"][torch.arange(total_batch_size, device=device)[:, None], perms]

            ### rebatch for training
            samples_batched = {}
            for k, v in samples.items():
                if k in ["prompts", "prompt_metadata"]:
                    samples_batched[k] = [v[i:i + config['train']['batch_size']]
                                for i in range(0, len(v), config['train']['batch_size'])]
                elif k in ["unet_outputs"]:
                    samples_batched[k] = v.reshape(-1, config['train']['batch_size'], *v.shape[1:])
                else:
                    samples_batched[k] = v.reshape(-1, config['train']['batch_size'], *v.shape[1:])

            # dict of lists -> list of dicts for easier iteration
            samples_batched = [
                dict(zip(samples_batched, x)) for x in zip(*samples_batched.values())
            ] # len = sample_bs * num_batches_per_epoch // train_bs = num_train_batches_per_epoch

            unet.train()
            flow_model.train()
            info = defaultdict(list)
            for i, sample in tqdm(
                list(enumerate(samples_batched)),
                desc=f"Epoch {epoch}.{inner_epoch}: training",
                position=0,
                disable=not is_local_main_process,
            ):
                """
                sample: [
                ('prompts', list of strings, len=train_bs), ('prompt_metadata', list of dicts),
                (bf16) ('prompt_embeds', torch.Size([1, 77, 768])), 
                (int64) ('timesteps', torch.Size([1, 50])), 
                (bf16) ('latents', torch.Size([1, 50, 4, 64, 64])), ('next_latents', torch.Size([1, 50, 4, 64, 64])), 
                ('log_probs', torch.Size([1, 50])), 
                ]
                """
                if config['train']['cfg']:
                    # concat negative prompts to sample prompts to avoid two forward passes
                    embeds = torch.cat(
                        [train_neg_prompt_embeds, sample["prompt_embeds"]]
                    )
                else:
                    embeds = sample["prompt_embeds"]

                for j in tqdm(range(num_train_timesteps), desc="Timestep", position=1, leave=False, disable=not is_local_main_process):
                    with autocast():
                        if config['train']['cfg']:
                            noise_pred = unet(
                                torch.cat([sample["latents"][:, j]] * 2),
                                torch.cat([sample["timesteps"][:, j]] * 2),
                                embeds,
                            ).sample
                            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                            noise_pred = (
                                    noise_pred_uncond
                                    + config['sample']['guidance_scale']
                                    * (noise_pred_text - noise_pred_uncond)
                            )
                            if config['train']['unetreg'] > 0:
                                unetreg = (noise_pred - sample["unet_outputs"][:, j])**2
                                unetreg = torch.mean(unetreg, dim=(1, 2, 3))

                        else:
                            noise_pred = unet(
                                sample["latents"][:, j],
                                sample["timesteps"][:, j],
                                embeds,
                            ).sample   # (bs, 4, 64, 64)
                            if config['train']['unetreg'] > 0:
                                unetreg = (noise_pred - sample["unet_outputs"][:, j])**2

                        _, log_pf, log_pb = ddim_step_with_logprob(
                            pipeline.scheduler, noise_pred,
                            sample["timesteps"][:, j], # (train_bs, 50) -> (train_bs,)
                            sample["latents"][:, j], eta=config['sample']['eta'],
                            prev_sample=sample["next_latents"][:, j], calculate_pb=True,
                        ) # log_pf :(bs,)

                    #######################################################
                    #################### GFN ALGORITHM ####################
                    #######################################################
                    with autocast_flow():
                        flow = flow_model(sample["latents"][:, j], sample["timesteps"][:, j], sample["prompt_embeds"])
                        timestep_next = torch.clamp(sample["timesteps"][:, j] - scheduler_dt, min=0)
                        flow_next = flow_model(sample["next_latents"][:, j], timestep_next, sample["prompt_embeds"])

                    with autocast(), torch.no_grad():
                        unet_output = unet(sample["latents"][:, j], sample["timesteps"][:, j], sample["prompt_embeds"]).sample
                        latent = pred_orig_latent(pipeline.scheduler, unet_output, sample["latents"][:, j], sample["timesteps"][:, j])
                    with torch.inference_mode():
                        logr_tmp = reward_fn(decode(latent), sample["prompts"], sample["prompt_metadata"])[0]   # tuple -> tensor
                    logr = reward_transform(logr_tmp)
                    flow = flow + logr # bf16 + float32 -> float32

                    with autocast(), torch.no_grad():
                        unet_output = unet(sample["next_latents"][:, j], timestep_next, sample["prompt_embeds"]).sample
                        latent_next = pred_orig_latent(pipeline.scheduler, unet_output, sample["next_latents"][:, j], timestep_next)
                    with torch.inference_mode():
                        logr_next_tmp = reward_fn(decode(latent_next), sample["prompts"], sample["prompt_metadata"])[0]
                    logr_next = reward_transform(logr_next_tmp)
                    flow_next = flow_next + logr_next
                    end_mask = sample["timesteps"][:, j] == pipeline.scheduler.timesteps[-1] # RHS is 1
                    flow_next[end_mask] = reward_transform(sample['rewards'][end_mask].to(flow_next))

                    info["log_pf"].append(torch.mean(log_pf).detach())
                    info["flow"].append(torch.mean(flow).detach())
                    info["log_pb"].append(torch.mean(log_pb).detach())

                    if config['train']['klpf'] > 0:
                        losses_flow = (flow + log_pf.detach() - log_pb.detach() - flow_next) ** 2

                        flow_next_klpf = flow_next.detach()
                        log_pb_klpf, log_pf_klpf = log_pb.detach(), log_pf.detach()
                        reward_db = (flow_next_klpf + log_pb_klpf - log_pf_klpf - flow).detach()

                        # different gpu has different states, so cannot share a baseline
                        assert len(reward_db) > 1
                        rloo_baseline = (reward_db.sum() - reward_db) / (len(reward_db) - 1)
                        reward_ = (reward_db - rloo_baseline) ** 2
                        rloo_var = (reward_.sum() - reward_) / (len(reward_db) - 1)
                        advantages = (reward_db - rloo_baseline) / (rloo_var.sqrt() + 1e-8)
                        advantages = torch.clamp(advantages, -config['train']['adv_clip_max'], config['train']['adv_clip_max'])

                        ratio = torch.exp(log_pf - sample["log_probs"][:, j])
                        unclipped_losses = -advantages * ratio
                        clipped_losses = -advantages * torch.clamp(
                            ratio,
                            1.0 - config['train']['clip_range'],
                            1.0 + config['train']['clip_range'],
                            )
                        losses_klpf = torch.maximum(unclipped_losses, clipped_losses)
                        info["ratio"].append(torch.mean(ratio).detach())

                        losses = losses_flow + config['train']['klpf'] * losses_klpf
                        info["loss"].append(losses_flow.mean().detach())
                        info["loss_klpf"].append(losses_klpf.mean().detach())
                        torch.cuda.empty_cache() # clear comp graph for log_pf_next
                    else:
                        losses_gfn = (flow + log_pf - log_pb - flow_next) ** 2  # (bs,)
                        info["loss"].append(losses_gfn.mean().detach())
                        losses = losses_gfn

                    if config['train']['unetreg'] > 0:
                        losses = losses + config['train']['unetreg'] * unetreg
                        info["unetreg"].append(unetreg.mean().detach())
                    loss = torch.mean(losses)

                    if logr_tmp is not None:
                        info["logr"].append(torch.mean(logr_tmp).detach())

                    loss = loss / accumulation_steps
                    if scaler:
                        # Backward passes under autocast are not recommended
                        scaler.scale(loss).backward()
                    else:
                        loss.backward()

                    # prevent OOM
                    image_next = image = prev_sample_klpf = unet_output = latent = latent_next = latent_next_next = None
                    noise_pred_next_uncond = noise_pred_next_text = noise_pred_uncond = noise_pred_text = noise_pred = noise_pred_next = None
                    flow = flow_next = flow_next_next = logr = logr_next = logr_next_next = logr_next_tmp = logr_tmp = reward_db = advantages = None
                    _ = log_pf = log_pb = log_pf_next = log_pb_next = log_pf_klpf = log_pb_klpf = None
                    unetreg = unetreg_initial = losses = losses_flow = losses_klpf = losses_gfn = None

                if ((j == num_train_timesteps - 1) and
                        (i + 1) % config['train']['gradient_accumulation_steps'] == 0):
                    if scaler:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(unet.parameters(), config['train']['max_grad_norm'])
                        torch.nn.utils.clip_grad_norm_(flow_model.parameters(), config['train']['max_grad_norm'])
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(unet.parameters(), config['train']['max_grad_norm'])
                        torch.nn.utils.clip_grad_norm_(flow_model.parameters(), config['train']['max_grad_norm'])
                        optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1

                    info = {k: torch.mean(torch.stack(v)) for k, v in info.items()}
                    dist.barrier()
                    for k, v in info.items():
                        dist.all_reduce(v, op=dist.ReduceOp.SUM)
                    info = {k: v / num_processes for k, v in info.items()}
                    for k, v in info.items():
                        result[k][global_step] = v.item()

                    info.update({"epoch": epoch})
                    result["epoch"][global_step] = epoch
                    result["time"][global_step] = time.time() - start_time

                    if is_local_main_process:
                        if config['wandb']:
                            wandb.log(info, step=global_step)
                        logger.info(f"global_step={global_step}  " +
                              " ".join([f"{k}={v:.3f}" for k, v in info.items()]))
                    info = defaultdict(list) # reset info dict

        if is_local_main_process:
            pickle.dump(result, gzip.open(os.path.join(output_dir, f"result.json"), 'wb'))
        dist.barrier()

        if epoch % config['save_freq'] == 0 or epoch == config['num_epochs'] - 1:
            if is_local_main_process:
                save_path = os.path.join(output_dir, f"checkpoint_epoch{epoch}")
                unwrapped_unet = unwrap_model(unet)
                unet_lora_state_dict = convert_state_dict_to_diffusers(
                    get_peft_model_state_dict(unwrapped_unet)
                )
                StableDiffusionPipeline.save_lora_weights(
                    save_directory=save_path,
                    unet_lora_layers=unet_lora_state_dict,
                    is_main_process=is_local_main_process,
                    safe_serialization=True,
                )
                logger.info(f"Saved state to {save_path}")

            dist.barrier()

    if config['wandb'] and is_local_main_process:
        wandb.finish()


if __name__ == "__main__":
    main()
    dist.destroy_process_group()
