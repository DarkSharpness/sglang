
from functools import lru_cache
from typing import Tuple

from sglang.srt.distributed import get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size


@lru_cache
def get_tp_info() -> Tuple[int, int]:
    return get_tensor_model_parallel_rank(), get_tensor_model_parallel_world_size()


def divide_uneven_by_head(num: int, head: int, layer: int) -> int:
    assert num % head == 0
    rank, size = get_tp_info()
    partition = head // size
    if (rank + layer) % size < head % size:
        partition += 1
    return (num // head) * partition
