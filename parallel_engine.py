"""Training and evaluation steps for a (pp, dp, tp) layout, built on PyTorch's pipelining,
FSDP2 and DTensor. The single-device `engine.py` stays the reference these must match."""

from collections import defaultdict
import contextlib
import math

import torch
import torch.distributed as dist
from torch.distributed.pipelining import PipelineStage, Schedule1F1B, ScheduleGPipe
from torch.distributed.tensor import DTensor, Shard
from torch.distributed.tensor.parallel import loss_parallel
from torch.nn import functional as F

from data import shard_global_batch
from engine import next_token_labels

SCHEDULES = {"1f1b": Schedule1F1B, "gpipe": ScheduleGPipe}


class ParallelEngine:
    def __init__(self, stage, ctx, batch_size, microbatches=1, schedule="1f1b"):
        if ctx.pp > 1 and microbatches < ctx.pp and schedule == "1f1b":
            raise ValueError(f"1F1B needs --microbatches >= --pp ({ctx.pp}); got {microbatches}")
        self.stage, self.ctx, self.microbatches, self.batch_size = stage, ctx, microbatches, batch_size
        # Loss is a SUM divided by this count, set per step (see _set_normalizer).
        self.divisor = 1.0
        self.schedule = self.eval_schedule = None
        if ctx.pp > 1:
            self.schedule = self._pipeline(schedule)
            # Evaluation gets its own stage state. Some PyTorch releases (2.11 on Context3) cannot
            # switch one pipeline stage from forward-only to training: its receive buffers stay
            # without requires_grad, and the first backward finds "gradients None".
            self.eval_schedule = self._pipeline(schedule)
        self.tied = stage.config.tie_word_embeddings and ctx.pp > 1

    def train_step(self, global_batch, optimizer, max_grad_norm):
        self.stage.train()
        optimizer.zero_grad(set_to_none=True)
        local, labels, count = self._prepare(global_batch)
        # dp ranks are averaged by the data-parallel wrapper, so each contributes sum/(count/dp).
        self.divisor = count / self.ctx.dp
        losses = self._run(local, labels, training=True)
        loss = self._share_loss(losses)
        if self.ctx.dp > 1:
            dist.all_reduce(loss, op=dist.ReduceOp.AVG, group=self.ctx.dp_group)
        if not torch.isfinite(loss).item():
            raise RuntimeError("Non-finite training loss")
        self._align_gradients()
        self._sync_tied_gradients()
        grad_norm = self._clip(max_grad_norm)
        optimizer.step()
        return loss.item(), count, grad_norm

    # Mean loss per valid target token over the whole (global) validation set.
    @torch.no_grad()
    def evaluate(self, data, batch_size):
        if batch_size > self.batch_size:
            raise ValueError("Evaluation batch size cannot exceed the training batch size")
        self.stage.eval()
        total = torch.zeros(2, device=self.ctx.device, dtype=torch.float64)
        try:
            for start in range(0, len(data["input_ids"]), batch_size):
                batch = {name: tensor[start:start + batch_size] for name, tensor in data.items()}
                local, labels, count = self._prepare(batch)
                self.divisor = 1.0  # plain sums; normalized once below
                losses = self._run(local, labels, training=False)
                total[0] += self._share_loss(losses).double()
                total[1] += count
        finally:
            self.stage.train()
        # Each dp rank scored a different slice of every batch; token counts are already global.
        if self.ctx.dp > 1:
            dist.all_reduce(total[:1], group=self.ctx.dp_group)
        return (total[0] / total[1]).item()

    # Every rank derives the same global count from the same global batch: no collective needed.
    def _prepare(self, global_batch):
        labels = next_token_labels(global_batch)
        count = int((labels != -100).sum().item())
        if count == 0:
            raise ValueError("Batch has no valid next-token targets")
        local = shard_global_batch(global_batch, self.ctx.dp_rank, self.ctx.dp,
                                   self.microbatches, self.batch_size)
        local = {name: tensor.to(self.ctx.device) for name, tensor in local.items()}
        return local, next_token_labels(local), count

    # Run forward (and backward when training); the last stage returns microbatch losses.
    def _run(self, local, labels, training):
        ctx = self.ctx
        losses = []
        if self.schedule is not None:
            run = self.schedule.step if training else self.eval_schedule.eval
            with self._context():
                if ctx.is_first_stage and ctx.is_last_stage:
                    run(local["input_ids"], target=labels, losses=losses)
                elif ctx.is_first_stage:
                    run(local["input_ids"])
                elif ctx.is_last_stage:
                    run(target=labels, losses=losses)
                else:
                    run()
        else:
            for ids, chunk in zip(local["input_ids"].chunk(self.microbatches),
                                  labels.chunk(self.microbatches)):
                with self._context():
                    loss = self._loss(self.stage(ids), chunk)
                    if training:
                        loss.backward()
                losses.append(loss.detach())
        return [loss.full_tensor() if isinstance(loss, DTensor) else loss for loss in losses]

    def _pipeline(self, schedule):
        stage = PipelineStage(self.stage, self.ctx.pp_rank, self.ctx.pp, self.ctx.device,
                              group=self.ctx.pp_group)
        return SCHEDULES[schedule](stage, n_microbatches=self.microbatches, loss_fn=self._loss,
                                   scale_grads=False)

    # Sum of token losses over one microbatch, divided by the divisor. FSDP2 warns that the
    # last stage's logits are a view; this only reshapes and reads them (nothing in-place).
    def _loss(self, logits, labels):
        flat = logits.reshape(-1, logits.shape[-1])
        if self.ctx.tp > 1:
            # Logits arrive split over the vocabulary; wrap them so the loss is vocabulary-parallel.
            flat = DTensor.from_local(flat, self.ctx.mesh["tp"], [Shard(1)])
        loss = F.cross_entropy(flat, labels.reshape(-1), ignore_index=-100, reduction="sum")
        return loss / self.divisor

    def _context(self):
        return loss_parallel() if self.ctx.tp > 1 else contextlib.nullcontext()

    # Add the per-microbatch losses on the last stage, then share the total with every rank.
    def _share_loss(self, losses):
        total = torch.zeros((), device=self.ctx.device)
        for loss in losses:
            total += loss.detach().to(total)
        if self.ctx.pp > 1:
            dist.all_reduce(total, group=self.ctx.pp_group)
        return total

    # Autograd can hand back a DTensor gradient laid out differently from its parameter (for
    # example a full Replicate gradient for a vocabulary-sharded embedding). Convert each to
    # the parameter's own layout: a local slice, or a reduction if the gradient is Partial.
    def _align_gradients(self):
        for name, parameter in self.stage.named_parameters():
            grad = parameter.grad
            if parameter.requires_grad and grad is None:
                raise RuntimeError(f"Missing gradient: {name}")
            if isinstance(grad, DTensor) and tuple(grad.placements) != tuple(parameter.placements):
                parameter.grad = grad.redistribute(grad.device_mesh, parameter.placements)

    # The embedding is stored on the first and last stages; each saw only part of its gradient.
    def _sync_tied_gradients(self):
        if not self.tied or not (self.ctx.is_first_stage or self.ctx.is_last_stage):
            return  # middle stages hold neither end of the tied matrix
        weight = self.stage.model.embed_tokens.weight if self.ctx.is_first_stage else self.stage.lm_head.weight
        if weight.grad is None:
            raise RuntimeError("Missing gradient for the tied embedding")
        grad = weight.grad.to_local() if isinstance(weight.grad, DTensor) else weight.grad
        dist.all_reduce(grad, group=self.ctx.tied_group)

    # Gradient norm over ALL parameters of ALL stages, counting a tied matrix once.
    def _clip(self, max_norm):
        ctx, stage = self.ctx, self.stage
        duplicate = stage.lm_head.weight if (self.tied and ctx.is_last_stage) else None
        # Group by mesh: DTensors on different meshes (and plain tensors) cannot be stacked.
        groups, counted = defaultdict(list), defaultdict(list)
        for name, parameter in stage.named_parameters():
            if parameter.requires_grad and parameter.grad is None:
                raise RuntimeError(f"Missing gradient: {name}")
            key = parameter.grad.device_mesh if isinstance(parameter.grad, DTensor) else None
            groups[key].append(parameter)
            if parameter is not duplicate:
                counted[key].append(parameter.grad)
        squared = torch.zeros((), device=ctx.device)
        for grads in counted.values():
            norm = torch.nn.utils.get_total_norm(grads, 2.0, error_if_nonfinite=True)
            squared += (norm.full_tensor() if isinstance(norm, DTensor) else norm).to(squared) ** 2
        if ctx.pp > 1:
            dist.all_reduce(squared, group=ctx.pp_group)
        total = squared.sqrt()
        if not math.isfinite(total.item()):
            raise RuntimeError("Non-finite gradient norm")
        for parameters in groups.values():
            torch.nn.utils.clip_grads_with_norm_(parameters, max_norm, total)
        return total.item()
