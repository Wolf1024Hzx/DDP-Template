"""
@Author: hzx
@Date: 2025-04-25
@Version: 1.0
"""

import os

import torch.distributed as dist


def is_main_process() -> bool:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank() == 0
    else:
        local_rank = os.getenv("LOCAL_RANK", "0")
        rank = os.getenv("RANK", "0")
        return local_rank == "0" or rank == "0"
