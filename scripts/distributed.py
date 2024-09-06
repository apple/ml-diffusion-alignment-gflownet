# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.

import datetime
import os
import logging
import torch
import torch.distributed as dist
import irisctl.api as irisctl


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def init_distributed_multinode(timeout=0):
    master_host = ""
    world_size = 0
    for tasklet in irisctl.distributed_tasklets():
        if tasklet.role_rank == 0:
            master_host = f"{tasklet.host_ip_address}:{tasklet.distributed_port}"
        world_size += 1
    print(
        f"Init PyTorch DDP with master host {master_host}, "
        f"world size {world_size}, rank {irisctl.role_rank()}"
    )
    if timeout == 0:
        timeout = dist.default_pg_timeout
    else:
        timeout = datetime.timedelta(seconds=timeout)

    logging.info(f'Default timeout: {timeout}')
    if world_size >= 1:
        torch.distributed.init_process_group(
            backend="nccl",
            init_method="tcp://" + master_host,
            world_size=world_size,
            timeout=timeout,
            rank=irisctl.role_rank(),
        )

    logging.info("Starting {} workers with rank {}".format(world_size, irisctl.role_rank()))
    # Pick a GPU based on the local rank
    torch.cuda.set_device(irisctl.local_rank())

    dist.barrier()
    setup_for_distributed(irisctl.local_rank() == 0)
    return irisctl.local_rank(), irisctl.role_rank(), world_size


def init_distributed_singlenode(timeout=0):
    # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
    dist_url = "env://"  # default

    # only works with torch.distributed.launch // torch.run
    rank = int(os.environ["RANK"])
    world_size = int(os.environ['WORLD_SIZE'])
    local_rank = int(os.environ['LOCAL_RANK'])

    if timeout == 0:
        timeout = dist.default_pg_timeout
    else:
        timeout = datetime.timedelta(seconds=timeout)

    logging.info(f'Default timeout: {timeout}')
    dist.init_process_group(
        backend="nccl",
        init_method=dist_url,
        world_size=world_size,
        timeout=timeout,
        rank=rank)

    # this will make all .cuda() calls work properly
    torch.cuda.set_device(local_rank)
    # synchronizes all the threads to reach this point before moving on
    dist.barrier()
    logging.info(f'setting up local_rank {local_rank} global_rank {rank} world size {world_size}')
    setup_for_distributed(rank == 0)
    return local_rank, rank, world_size


def get_rank():
    return torch.distributed.get_rank() if torch.distributed.is_initialized() else 0


def get_local_rank():
    return int(os.environ.get('LOCAL_RANK', '0'))


# ----------------------------------------------------------------------------

def get_world_size():
    return torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1


def print0(*args, **kwargs):
    if get_rank() == 0:
        print(*args, **kwargs)


def set_seed(seed):
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    import numpy as np
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.random.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.empty_cache()

    logging.info(f'Using seed: {seed}')


def load_distributed(ddp_model, CHECKPOINT_PATH, rank=0):
    # configure map_location properly
    map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
    # ddp_model.load_attn_procs( # ?
    ddp_model.load_state_dict(
        torch.load(CHECKPOINT_PATH, map_location=map_location))
