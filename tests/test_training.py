"""Offline CPU integration tests against real HF Llama layers, not the old toy model."""

import copy
import tempfile
import unittest

import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

from data import fingerprint, pack_texts, training_batch
from engine import evaluate, next_token_loss, restore_training, save_checkpoint, train_step
from inference import generate_text
from model import load_model


class TrainingTests(unittest.TestCase):
    def test_full_training_resume_and_generation(self):
        torch.manual_seed(42)
        torch.set_num_threads(1)
        tokenizer = make_tokenizer()
        config = LlamaConfig(vocab_size=len(tokenizer), hidden_size=16, intermediate_size=32,
                             num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                             max_position_embeddings=64, bos_token_id=0, eos_token_id=0,
                             pad_token_id=0, attention_dropout=0.1, tie_word_embeddings=True)
        config._attn_implementation = "eager"
        model = LlamaForCausalLM(config)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        data = pack_texts(["one two three four five six seven eight"] * 4, tokenizer, 8)
        batch = training_batch(data, 0, 2, 42, torch.device("cpu"))
        before = {n: p.detach().clone() for n, p in model.named_parameters()}
        loss, tokens, norm = train_step(model, batch, optimizer, 1.0)
        self.assertGreater(tokens, 0)
        self.assertGreater(norm, 0)
        self.assertTrue(all(p.requires_grad and p.grad is not None and torch.isfinite(p.grad).all()
                            for p in model.parameters()))
        self.assertTrue(all(not torch.equal(before[n], p) for n, p in model.named_parameters()))
        settings = {"train_data": fingerprint(data)}
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = save_checkpoint(model, tokenizer, optimizer, 1, settings, directory)
            with self.assertRaises(FileExistsError):
                save_checkpoint(model, tokenizer, optimizer, 1, settings, directory)
            next_batch = training_batch(data, 1, 2, 42, torch.device("cpu"))
            # Reference: continue uninterrupted, including dropout RNG and AdamW state.
            expected_loss, _, _ = train_step(model, next_batch, optimizer, 1.0)
            expected = copy.deepcopy(model.state_dict())
            restored, restored_tokenizer = load_model(str(checkpoint), torch.device("cpu"))
            resumed_optimizer = torch.optim.AdamW(restored.parameters(), lr=1e-3)
            step = restore_training(checkpoint, resumed_optimizer, settings, torch.device("cpu"))
            self.assertEqual(step, 1)
            actual_loss, _, _ = train_step(restored, next_batch, resumed_optimizer, 1.0)
            self.assertEqual(expected_loss, actual_loss)
            for name, value in restored.state_dict().items():
                torch.testing.assert_close(value, expected[name], rtol=0, atol=0)
            self.assertTrue(torch.isfinite(torch.tensor(evaluate(restored, data, 3, torch.device("cpu")))))
            self.assertTrue(restored.training)
            # Token-weighted validation must be independent of batch partitioning.
            self.assertAlmostEqual(evaluate(restored, data, 1, torch.device("cpu")),
                                   evaluate(restored, data, 3, torch.device("cpu")), places=5)
            text = generate_text(restored, restored_tokenizer, "one two", torch.device("cpu"), 3)
            self.assertTrue(text.startswith("one two"))
            with self.assertRaises(ValueError):
                restore_training(checkpoint, resumed_optimizer, {}, torch.device("cpu"))

    def test_shift_and_padding_preserve_real_eos(self):
        batch = {"input_ids": torch.tensor([[1, 2, 0, 0]]),
                 "attention_mask": torch.tensor([[1, 1, 1, 0]])}
        logits = torch.randn(1, 4, 4, requires_grad=True)
        loss, count = next_token_loss(logits, batch)
        expected = torch.nn.functional.cross_entropy(logits[0, :2], torch.tensor([2, 0]), reduction="sum")
        torch.testing.assert_close(loss, expected)
        self.assertEqual(count, 2)
        loss.backward()
        self.assertTrue(torch.equal(logits.grad[0, 2:], torch.zeros(2, 4)))

    def test_packing_and_epoch_cursor(self):
        tokenizer = make_tokenizer()
        data = pack_texts(["one two three four five six seven eight"] * 3, tokenizer, 8)
        self.assertEqual(int(data["attention_mask"][-1].sum()), 3)
        first = training_batch(data, 0, 2, 42, torch.device("cpu"))
        repeated = training_batch(data, 0, 2, 42, torch.device("cpu"))
        self.assertTrue(torch.equal(first["input_ids"], repeated["input_ids"]))
        self.assertEqual(len(training_batch(data, 1, 3, 42, torch.device("cpu"))["input_ids"]), 1)
        with self.assertRaises(ValueError):
            pack_texts([""], tokenizer, 8)


# Tiny local tokenizer makes the tests independent of downloads/authentication.
def make_tokenizer():
    backend = Tokenizer(WordLevel({"<eos>": 0, "<unk>": 1, "one": 2, "two": 3,
                                   "three": 4, "four": 5, "five": 6, "six": 7,
                                   "seven": 8, "eight": 9}, unk_token="<unk>"))
    backend.pre_tokenizer = Whitespace()
    return PreTrainedTokenizerFast(tokenizer_object=backend, eos_token="<eos>",
                                  pad_token="<eos>", unk_token="<unk>")


if __name__ == "__main__":
    unittest.main()
