"""Process mesh for pipeline (pp) x data (dp) x tensor (tp) parallelism.

Rank layout is rank = pp_index * (dp * tp) + dp_index * tp + tp_index, so tensor
parallel peers are adjacent ranks (the fastest links) and pipeline stages are farthest apart.
"""

from dataclasses import dataclass
import os
import sys
import traceback

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh


@dataclass
class ParallelContext:
    dp: int
    tp: int
    pp: int
    rank: int
    world_size: int
    device: torch.device
    mesh: object
    pp_rank: int
    dp_rank: int
    tp_rank: int
    tied_group: object  # first<->last stage peers sharing one embedding matrix; None if pp == 1

    @property
    def is_first_stage(self):
        return self.pp_rank == 0

    @property
    def is_last_stage(self):
        return self.pp_rank == self.pp - 1

    @property
    def is_main(self):
        return self.rank == 0

    @property
    def pp_group(self):
        return self.mesh["pp"].get_group()

    @property
    def dp_group(self):
        return self.mesh["dp"].get_group()

    @property
    def tp_group(self):
        return self.mesh["tp"].get_group()

    @property
    def layout(self):
        return {"dp": self.dp, "tp": self.tp, "pp": self.pp}


# Join the job described by torchrun's environment and build the (pp, dp, tp) mesh.
def init_parallel(dp, tp, pp, device_type):
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    if device_type == "cuda":
        local = int(os.environ.get("LOCAL_RANK", rank))
        torch.cuda.set_device(local)
        device = torch.device("cuda", local)
        backend = "nccl"
    elif device_type == "cpu":
        device, backend = torch.device("cpu"), "gloo"
    else:
        raise ValueError("Parallel execution supports --device cuda (NCCL) or cpu (gloo)")
    if not dist.is_initialized():
        dist.init_process_group(backend, device_id=device if device_type == "cuda" else None)
    mesh = init_device_mesh(device_type, (pp, dp, tp), mesh_dim_names=("pp", "dp", "tp"))
    coordinate = mesh.get_coordinate()
    tied_group = None
    if pp > 1:
        # new_group is collective: every rank creates every group, in the same order.
        for dp_index in range(dp):
            for tp_index in range(tp):
                members = [dp_index * tp + tp_index, (pp - 1) * dp * tp + dp_index * tp + tp_index]
                group = dist.new_group(members)
                if rank in members:
                    tied_group = group
    return ParallelContext(dp, tp, pp, rank, world_size, device, mesh,
                           coordinate[0], coordinate[1], coordinate[2], tied_group)


# Reject layouts the model's shapes cannot be divided into, before any process allocates.
def validate_layout(config, dp, tp, pp, world_size):
    if min(dp, tp, pp) < 1:
        raise ValueError("--dp, --tp and --pp must be positive")
    if dp * tp * pp != world_size:
        raise ValueError(f"dp*tp*pp = {dp * tp * pp} but the job has {world_size} processes")
    if pp > config.num_hidden_layers:
        raise ValueError(f"--pp {pp} exceeds the model's {config.num_hidden_layers} layers")
    for name, value in (("attention heads", config.num_attention_heads),
                        ("key/value heads", config.num_key_value_heads),
                        ("intermediate size", config.intermediate_size),
                        ("vocabulary size", config.vocab_size)):
        if value % tp:
            raise ValueError(f"--tp {tp} does not divide the model's {name} ({value})")


# Run `work()`, then shut down cleanly. A rank that fails must say why and exit AT ONCE:
# waiting for its peers in a collective (as a clean shutdown does) hangs until NCCL's
# 10-minute watchdog fires and hides the original traceback. torchrun then stops the others.
def run_or_exit(work):
    try:
        work()
    except BaseException:
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)
    shutdown()


# Run `work` on rank 0 first (downloads, dataset caches), then on everyone else.
def rank_zero_first(ctx, work):
    result = work() if ctx.is_main else None
    dist.barrier()
    return result if ctx.is_main else work()


def shutdown():
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
