"""T6 distillation: leakage-safe sampling, dataset format, masking, agreement, and a
real LoRA training run on a tiny model built locally (skipped without the `ml` extra)."""

import json

import numpy as np
import pandas as pd
import pytest

from newsalpha.config import ExtractionConfig
from newsalpha.distill.dataset import read_jsonl, sample_for_teacher, teacher_records, write_splits
from newsalpha.distill.evaluate import agreement
from newsalpha.extraction.rules import RulesExtractor
from newsalpha.extraction.schema import RawExtraction

CUTOFF = "2024-02-15"


@pytest.fixture(scope="module")
def splits(tmp_path_factory, articles):
    sample = sample_for_teacher(articles, n=60, before=CUTOFF, seed=0)
    recs = teacher_records(sample, RulesExtractor(), ExtractionConfig())
    out = tmp_path_factory.mktemp("distill")
    meta = write_splits(recs, out, val_frac=0.2)
    return sample, out, meta


def test_sample_is_training_period_only_and_covers_every_month(splits, articles):
    sample, _, _ = splits
    assert (sample["created_at"] < pd.Timestamp(CUTOFF, tz="UTC")).all()
    early = articles[articles["created_at"] < pd.Timestamp(CUTOFF, tz="UTC")]
    assert set(sample["created_at"].dt.month) == set(early["created_at"].dt.month)


def test_validation_is_later_than_training(splits):
    _, out, meta = splits
    train, val = read_jsonl(out / "train.jsonl"), read_jsonl(out / "val.jsonl")
    assert len(train) == meta["train"] and len(val) == meta["val"] > 0
    assert max(r["created_at"] for r in train) <= min(r["created_at"] for r in val)
    assert not {r["article_id"] for r in train} & {r["article_id"] for r in val}


def test_records_are_chat_examples_with_a_valid_answer(splits):
    _, out, _ = splits
    rec = read_jsonl(out / "train.jsonl")[0]
    assert [m["role"] for m in rec["messages"]] == ["system", "user", "assistant"]
    assert rec["messages"][1]["content"].startswith("HEADLINE:")
    RawExtraction.model_validate(json.loads(rec["messages"][2]["content"]))


def test_agreement_table():
    t = pd.DataFrame({"article_id": ["a", "b"], "event_type": ["earnings", "other"], "direction": [1.0, 0.0],
                      "eps_surprise": [0.10, np.nan], "rev_surprise": np.nan, "rating_change": np.nan,
                      "target_change": np.nan, "guidance_change": np.nan})  # fmt: skip
    s = t.copy()
    s.loc[0, "eps_surprise"] = 0.105
    s.loc[1, "direction"] = 1.0
    a = agreement(t, s).set_index("field")
    assert a.loc["event_type", "agreement"] == 1.0 and a.loc["direction", "agreement"] == 0.5
    assert a.loc["eps_surprise", "agreement"] == 1.0 and a.loc["eps_surprise", "coverage"] == 1.0


@pytest.fixture(scope="module")
def tiny_model(tmp_path_factory, splits):
    """A 1-layer Llama with a word-level tokenizer trained on the dataset: no downloads."""
    pytest.importorskip("torch")
    pytest.importorskip("peft")
    tokenizers = pytest.importorskip("tokenizers")
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

    _, out, _ = splits
    texts = [m["content"] for r in read_jsonl(out / "train.jsonl") for m in r["messages"]]
    tk = tokenizers.Tokenizer(tokenizers.models.WordLevel(unk_token="[UNK]"))
    tk.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
    tk.train_from_iterator(
        texts, tokenizers.trainers.WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "</s>"])
    )
    tok = PreTrainedTokenizerFast(
        tokenizer_object=tk, unk_token="[UNK]", pad_token="[PAD]", eos_token="</s>"
    )
    cfg = LlamaConfig(vocab_size=tk.get_vocab_size(), hidden_size=32, intermediate_size=64,
                      num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
                      max_position_embeddings=1024)  # fmt: skip
    path = tmp_path_factory.mktemp("tiny_base")
    LlamaForCausalLM(cfg).save_pretrained(path)
    tok.save_pretrained(path)
    return path, tok


def test_loss_is_on_the_answer_only(tiny_model, splits):
    from newsalpha.distill.train_lora import IGNORE, encode

    _, tok = tiny_model
    _, out, _ = splits
    enc = encode(tok, read_jsonl(out / "train.jsonl")[0]["messages"], max_len=4096)
    labels = enc["labels"]
    n_masked = sum(1 for x in labels if x == IGNORE)
    assert 0 < n_masked < len(labels)
    assert all(x == IGNORE for x in labels[:n_masked])  # prompt first, then answer


def test_lora_training_runs_end_to_end(tiny_model, splits, tmp_path):
    from newsalpha.distill.train_lora import train

    base, _ = tiny_model
    _, out, _ = splits
    meta = train(str(base), out, tmp_path / "adapter", r=4, alpha=8, max_len=512,
                 batch_size=2, grad_accum=1, max_steps=2)  # fmt: skip
    assert (tmp_path / "adapter" / "adapter_model.safetensors").exists()
    assert meta["n_train"] > 0 and np.isfinite(meta["train_loss"])
