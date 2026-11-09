"""Multi-process CPU (gloo) tests: every parallel layout must reproduce the single-device run.

Each test spawns real processes that join a real process group, shard a tiny HF Llama with
DTensor tensor parallelism, FSDP2 data parallelism and torch.distributed.pipelining, and
compare against engine.train_step / HF generate on the unsharded model. No downloads.
"""

import os
import socket
import tempfile
import unittest

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.tensor import DTensor
from transformers import LlamaConfig, LlamaForCausalLM

from data import shard_global_batch, training_batch
from distributed import validate_layout
from engine import evaluate, next_token_labels, next_token_loss, train_step
from test_training import make_tokenizer

BATCH, LR, SEED = 3, 1e-2, 42
PROMPTS = ["one two", "three", "four five six seven", "eight six one", "two two three four five"]


# ---- shared fixtures -------------------------------------------------------------------

def tiny_config(vocab=64, init=0.02):
    # No pad_token_id (DTensor embeddings do not support padding_idx), no dropout.
    return LlamaConfig(vocab_size=vocab, hidden_size=16, intermediate_size=32, num_hidden_layers=4,
                       num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=64,
                       tie_word_embeddings=True, attention_dropout=0.0, initializer_range=init,
                       eos_token_id=0, bos_token_id=None)


def make_checkpoint(directory, **kwargs):
    torch.manual_seed(0)
    LlamaForCausalLM(tiny_config(**kwargs)).save_pretrained(directory)
    return directory


# Seven sequences; two end in padding, so batches of three give 3, 3 and then 1 row.
def make_data():
    generator = torch.Generator().manual_seed(3)
    ids = torch.randint(0, 64, (7, 8), generator=generator)
    mask = torch.ones(7, 8, dtype=torch.long)
    mask[6, 3:] = 0
    mask[5, 6:] = 0
    return {"input_ids": ids, "attention_mask": mask}


def reference_run(checkpoint, steps):
    model = LlamaForCausalLM.from_pretrained(checkpoint, dtype=torch.float32, attn_implementation="eager")
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    data, results = make_data(), []
    for step in range(steps):
        batch = training_batch(data, step, BATCH, SEED, torch.device("cpu"))
        results.append(train_step(model, batch, optimizer, 1.0))
    return model, results


def reference_name(name):
    return name.replace("lm_head.weight", "model.embed_tokens.weight")  # the tied matrix is one parameter


def full(parameter):
    return parameter.full_tensor() if isinstance(parameter, DTensor) else parameter


def free_port():
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


# Run `target(rank, *args)` in world_size processes; any exception in a worker fails the test.
def launch(target, world_size, *args):
    mp.spawn(_entry, args=(target, world_size, free_port(), args), nprocs=world_size)


def _entry(rank, target, world_size, port, args):
    os.environ.update(RANK=str(rank), WORLD_SIZE=str(world_size), LOCAL_RANK=str(rank),
                      MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    torch.set_num_threads(1)
    try:
        target(rank, *args)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


# ---- workers ---------------------------------------------------------------------------

def train_worker(rank, layout, checkpoint, microbatches, shard, schedule):
    from distributed import init_parallel
    from parallel_engine import ParallelEngine
    from parallel_model import create_stage
    ctx = init_parallel(*layout, "cpu")
    stage = create_stage(checkpoint, ctx, shard=shard)
    engine = ParallelEngine(stage, ctx, BATCH, microbatches, schedule)
    optimizer = torch.optim.AdamW(stage.parameters(), lr=LR, weight_decay=0.01, foreach=False)
    reference, expected = reference_run(checkpoint, 3)
    data = make_data()
    # Evaluate before training, as the driver does: a pipeline stage must survive eval -> train.
    initial = LlamaForCausalLM.from_pretrained(checkpoint, dtype=torch.float32, attn_implementation="eager")
    torch.testing.assert_close(engine.evaluate(data, BATCH),
                               evaluate(initial, data, BATCH, torch.device("cpu")), rtol=1e-5, atol=1e-6)
    for step in range(3):  # the last batch has 1 row: padding rows must add exactly zero
        batch = training_batch(data, step, BATCH, SEED, torch.device("cpu"))
        loss, tokens, norm = engine.train_step(batch, optimizer, 1.0)
        assert tokens == expected[step][1], (step, tokens, expected[step][1])
        torch.testing.assert_close(loss, expected[step][0], rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(norm, expected[step][2], rtol=1e-4, atol=1e-5)
    parameters = dict(reference.named_parameters())
    for name, parameter in stage.named_parameters():
        torch.testing.assert_close(full(parameter).detach(), parameters[reference_name(name)].detach(),
                                   rtol=0, atol=2e-6)
    # Validation is token-weighted over the global set whatever the layout or batch size.
    torch.testing.assert_close(engine.evaluate(data, BATCH),
                               evaluate(reference, data, BATCH, torch.device("cpu")), rtol=1e-5, atol=1e-6)


def save_worker(rank, layout, checkpoint, output, microbatches):
    from distributed import init_parallel
    from parallel_checkpoint import save_checkpoint
    from parallel_engine import ParallelEngine
    from parallel_model import create_stage
    ctx = init_parallel(*layout, "cpu")
    stage = create_stage(checkpoint, ctx)
    engine = ParallelEngine(stage, ctx, BATCH, microbatches)
    optimizer = torch.optim.AdamW(stage.parameters(), lr=LR, weight_decay=0.01, foreach=False)
    data = make_data()
    for step in range(2):
        engine.train_step(training_batch(data, step, BATCH, SEED, torch.device("cpu")), optimizer, 1.0)
    path = save_checkpoint(stage, make_tokenizer(), optimizer, 2, {"x": 1}, output, ctx)
    try:
        save_checkpoint(stage, make_tokenizer(), optimizer, 2, {"x": 1}, output, ctx)
    except FileExistsError:
        pass
    else:
        raise AssertionError("an existing checkpoint was overwritten")
    assert (path / "model.safetensors.index.json").exists() and (path / "training_state.pt").exists()
    assert not (path / "sharded").exists()


def resume_worker(rank, layout, checkpoint, resumed, microbatches):
    from distributed import init_parallel
    from parallel_checkpoint import restore_training
    from parallel_engine import ParallelEngine
    from parallel_model import create_stage
    ctx = init_parallel(*layout, "cpu")
    stage = create_stage(resumed, ctx)  # weights come from the checkpoint's Hugging Face files
    engine = ParallelEngine(stage, ctx, BATCH, microbatches)
    optimizer = torch.optim.AdamW(stage.parameters(), lr=LR, weight_decay=0.01, foreach=False)
    try:
        restore_training(resumed, stage, optimizer, {"x": 2}, ctx)
    except ValueError:
        pass
    else:
        raise AssertionError("resume accepted different settings")
    assert restore_training(resumed, stage, optimizer, {"x": 1}, ctx) == 2
    reference, expected = reference_run(checkpoint, 3)
    batch = training_batch(make_data(), 2, BATCH, SEED, torch.device("cpu"))
    loss, tokens, norm = engine.train_step(batch, optimizer, 1.0)
    torch.testing.assert_close(loss, expected[2][0], rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(norm, expected[2][2], rtol=1e-4, atol=1e-5)
    parameters = dict(reference.named_parameters())
    for name, parameter in stage.named_parameters():
        torch.testing.assert_close(full(parameter).detach(), parameters[reference_name(name)].detach(),
                                   rtol=0, atol=2e-6)


def generate_worker(rank, layout, checkpoint, microbatches, sample):
    from distributed import init_parallel
    from parallel_generate import generate_distributed
    from parallel_model import create_stage
    tokenizer = make_tokenizer()
    ctx = init_parallel(*layout, "cpu")
    model = create_stage(checkpoint, ctx, gather_logits=True, training=False, whole=layout[2] == 1)
    outputs = generate_distributed(model, tokenizer, PROMPTS, ctx, max_new_tokens=6, sample=sample,
                                   microbatches=microbatches)
    assert len(outputs) == len(PROMPTS)
    everyone = [None] * ctx.world_size
    dist.all_gather_object(everyone, outputs)
    assert all(other == outputs for other in everyone), "ranks disagree on the generated text"
    if sample:
        assert all(o.startswith(p) for o, p in zip(outputs, PROMPTS))
        return
    reference = LlamaForCausalLM.from_pretrained(checkpoint, attn_implementation="eager")
    for prompt, text in zip(PROMPTS, outputs):
        inputs = tokenizer(prompt, return_tensors="pt")
        expected = reference.generate(**inputs, max_new_tokens=6, do_sample=False, pad_token_id=0,
                                      eos_token_id=0)
        assert text == tokenizer.decode(expected[0], skip_special_tokens=True), (prompt, text)


# ---- tests -----------------------------------------------------------------------------

class LayoutTests(unittest.TestCase):
    def test_validate_layout(self):
        config = tiny_config()  # 4 heads, 2 KV heads, 4 layers, intermediate 32, vocab 64
        validate_layout(config, 2, 2, 2, 8)
        for dp, tp, pp, world in ((2, 2, 2, 4), (1, 3, 1, 3), (1, 4, 1, 4), (1, 1, 5, 5), (0, 1, 1, 0)):
            with self.assertRaises(ValueError, msg=(dp, tp, pp)):
                validate_layout(config, dp, tp, pp, world)
        # SmolLM2-135M: 9 query / 3 KV heads, 30 layers -> tp 3 works, tp 2 and tp 9 do not.
        smol = LlamaConfig(vocab_size=49152, hidden_size=576, intermediate_size=1536,
                           num_hidden_layers=30, num_attention_heads=9, num_key_value_heads=3)
        validate_layout(smol, 1, 3, 1, 3)
        for tp in (2, 9):
            with self.assertRaises(ValueError):
                validate_layout(smol, 1, tp, 1, tp)

    def test_global_batch_padding_is_exactly_neutral(self):
        data = make_data()
        batch = {name: tensor[:3] for name, tensor in data.items()}
        pieces = [shard_global_batch(batch, rank, 2, 2, BATCH) for rank in range(2)]
        self.assertTrue(all(len(p["input_ids"]) == 2 for p in pieces))  # 3 rows -> 4, split in two
        merged = {name: torch.cat([p[name] for p in pieces]) for name in batch}
        self.assertTrue(torch.equal(merged["input_ids"][:3], batch["input_ids"]))
        self.assertEqual(int(merged["attention_mask"][3:].sum()), 0)
        # Padding rows own no targets, so the total target count is unchanged.
        self.assertEqual(int((next_token_labels(merged) != -100).sum()),
                         int((next_token_labels(batch) != -100).sum()))
        with self.assertRaises(ValueError):
            shard_global_batch(data, 0, 1, 1, BATCH)

    def test_labels_match_the_single_device_loss(self):
        batch = {"input_ids": torch.tensor([[1, 2, 0, 0]]), "attention_mask": torch.tensor([[1, 1, 1, 0]])}
        self.assertEqual(next_token_labels(batch).tolist(), [[2, 0, -100, -100]])
        logits = torch.randn(1, 4, 5)
        loss, count = next_token_loss(logits, batch)
        expected = torch.nn.functional.cross_entropy(logits[0, :2], torch.tensor([2, 0]), reduction="sum")
        torch.testing.assert_close(loss, expected)
        self.assertEqual(count, 2)


class TrainingEquivalenceTests(unittest.TestCase):
    def check(self, dp, tp, pp, microbatches=1, shard=False, schedule="1f1b"):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as directory:
            launch(train_worker, dp * tp * pp, (dp, tp, pp), make_checkpoint(directory), microbatches,
                   shard, schedule)

    def test_data_parallel(self):
        self.check(2, 1, 1)

    def test_data_parallel_sharded_state(self):
        self.check(2, 1, 1, shard=True)

    def test_tensor_parallel(self):
        self.check(1, 2, 1)

    def test_pipeline_parallel_1f1b(self):
        self.check(1, 1, 3, microbatches=3)

    def test_pipeline_parallel_gpipe(self):
        self.check(1, 1, 2, microbatches=3, schedule="gpipe")

    def test_data_and_pipeline_sharded(self):
        self.check(2, 1, 2, microbatches=2, shard=True)

    def test_tensor_and_pipeline(self):
        self.check(1, 2, 2, microbatches=2)

    def test_all_three(self):
        self.check(2, 2, 2, microbatches=2)


class CheckpointTests(unittest.TestCase):
    def roundtrip(self, saved, resumed):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as directory:
            checkpoint = make_checkpoint(os.path.join(directory, "initial"))
            output = os.path.join(directory, "run")
            launch(save_worker, saved[0] * saved[1] * saved[2], saved, checkpoint, output, saved[2])
            step = os.path.join(output, "step-000002")
            # The export is an ordinary HF checkpoint equal to the single-device weights.
            exported = LlamaForCausalLM.from_pretrained(step)
            reference, _ = reference_run(checkpoint, 2)
            for name, value in reference.state_dict().items():
                torch.testing.assert_close(exported.state_dict()[name], value, rtol=0, atol=2e-6)
            launch(resume_worker, resumed[0] * resumed[1] * resumed[2], resumed, checkpoint, step, resumed[2])

    def test_resume_under_a_different_layout(self):
        self.roundtrip((2, 1, 2), (1, 2, 1))

    def test_resume_with_a_middle_pipeline_stage(self):
        # A middle stage holds neither end of the tied matrix, and each stage's parameter groups
        # must not be mixed up with the others' in the shared checkpoint.
        self.roundtrip((1, 1, 3), (1, 1, 3))

    def test_resume_from_three_stages_into_tensor_parallel(self):
        self.roundtrip((1, 1, 3), (1, 2, 1))

    def test_resume_from_one_stage_into_a_pipeline(self):
        # The tied matrix is one optimizer entry with pp == 1 and two with pp > 1.
        self.roundtrip((1, 2, 1), (1, 1, 2))


class GenerationTests(unittest.TestCase):
    def check(self, dp, tp, pp, microbatches=1, sample=False):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as directory:
            # A large initializer gives clear logit margins, so greedy choices are stable.
            checkpoint = make_checkpoint(directory, vocab=10, init=0.8)
            launch(generate_worker, dp * tp * pp, (dp, tp, pp), checkpoint, microbatches, sample)

    def test_single_process(self):
        self.check(1, 1, 1)

    def test_tensor_parallel(self):
        self.check(1, 2, 1)

    def test_data_parallel_uneven_split(self):
        self.check(3, 1, 1)  # 5 prompts over 3 replicas

    def test_pipeline_with_microbatches(self):
        self.check(1, 1, 4, microbatches=2)

    def test_all_three(self):
        self.check(2, 2, 2, microbatches=2)

    def test_sampling_is_consistent_across_ranks(self):
        self.check(1, 2, 2, microbatches=2, sample=True)


if __name__ == "__main__":
    unittest.main()
