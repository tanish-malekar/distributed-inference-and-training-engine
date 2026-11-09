"""Smallest possible NCCL test: all_reduce and barrier across the visible GPUs.

Separates cluster/NCCL problems from training-code problems. Launch under torchrun with one
process per GPU (scripts/context3.sh nccl GPUS does this after the GPU check and approval).
"""

import faulthandler
import os

import torch
import torch.distributed as dist


def main():
    faulthandler.dump_traceback_later(int(os.environ.get("DITE_DUMP_STACKS", "60")), exit=False)
    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    local = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local)
    device = torch.device("cuda", local)
    dist.init_process_group("nccl", device_id=device)
    say = lambda text: print(f"[rank {rank}] {text}", flush=True)
    say(f"initialized on {torch.cuda.get_device_name(device)}")
    value = torch.full((4,), float(rank + 1), device=device)
    dist.all_reduce(value)
    torch.cuda.synchronize(device)
    say(f"all_reduce small: {value.tolist()} (expected {sum(range(1, world + 1))})")
    dist.barrier()
    say("barrier ok")
    big = torch.ones(64 * 2**20 // 4, device=device)  # 64 MiB
    dist.all_reduce(big)
    torch.cuda.synchronize(device)
    say(f"all_reduce 64 MiB ok: {big[0].item()}")
    peer = (rank + 1) % world
    if world > 1:
        send = torch.full((8,), float(rank), device=device)
        receive = torch.empty(8, device=device)
        requests = dist.batch_isend_irecv([dist.P2POp(dist.isend, send, peer),
                                           dist.P2POp(dist.irecv, receive, (rank - 1) % world)])
        for request in requests:
            request.wait()
        torch.cuda.synchronize(device)
        say(f"send/recv ok: received {receive[0].item()} from rank {(rank - 1) % world}")
    dist.destroy_process_group()
    say("PASS")


if __name__ == "__main__":
    main()
