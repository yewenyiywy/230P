"""LoRA supervised fine-tuning of a small open model on teacher labels.

    python -m newsalpha.distill.train_lora \
        --base Qwen/Qwen2.5-7B-Instruct --data distill/ --out adapters/student_v1

Loss is computed on the assistant answer only (prompt tokens are masked with
-100), so the student learns to answer, not to repeat the article. Uses the
tokenizer's chat template when it has one. bf16 on GPU, fp32 on CPU.

Needs the `ml` extra (torch, transformers, peft). tests/test_distill.py runs
this end to end on a tiny randomly initialised model built locally.
"""

from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path
from typing import Any

from newsalpha.distill.dataset import read_jsonl

IGNORE = -100


def _ids(out) -> list[int]:
    """apply_chat_template returns a list or a dict-like BatchEncoding depending on version."""
    if isinstance(out, dict) or hasattr(out, "keys"):
        out = out["input_ids"]
    return list(out[0]) if out and isinstance(out[0], list) else list(out)


def encode(tokenizer, messages: list[dict], max_len: int) -> dict[str, list[int]]:
    prompt, answer = messages[:-1], messages[-1]["content"]
    if getattr(tokenizer, "chat_template", None):
        p_ids = _ids(
            tokenizer.apply_chat_template(prompt, add_generation_prompt=True, tokenize=True)
        )
        full = _ids(tokenizer.apply_chat_template(messages, tokenize=True))
    else:  # plain fallback template
        p_text = "".join(f"<|{m['role']}|>\n{m['content']}\n" for m in prompt) + "<|assistant|>\n"
        p_ids = tokenizer(p_text, add_special_tokens=False)["input_ids"]
        a_ids = tokenizer(answer, add_special_tokens=False)["input_ids"]
        full = (
            p_ids + a_ids + ([tokenizer.eos_token_id] if tokenizer.eos_token_id is not None else [])
        )
    full = full[:max_len]
    n_prompt = min(len(p_ids), len(full))
    labels = [IGNORE] * n_prompt + full[n_prompt:]
    return {"input_ids": full, "labels": labels}


class _Collator:
    def __init__(self, pad_id: int):
        self.pad_id = pad_id

    def __call__(self, batch: list[dict]):
        import torch

        n = max(len(b["input_ids"]) for b in batch)
        ids = [b["input_ids"] + [self.pad_id] * (n - len(b["input_ids"])) for b in batch]
        labels = [b["labels"] + [IGNORE] * (n - len(b["labels"])) for b in batch]
        mask = [[1] * len(b["input_ids"]) + [0] * (n - len(b["input_ids"])) for b in batch]
        return {
            "input_ids": torch.tensor(ids),
            "labels": torch.tensor(labels),
            "attention_mask": torch.tensor(mask),
        }


def train(
    base: str,
    data_dir: str | Path,
    out_dir: str | Path,
    r: int = 16,
    alpha: int = 32,
    epochs: float = 2.0,
    lr: float = 2e-4,
    max_len: int = 2048,
    batch_size: int = 4,
    grad_accum: int = 4,
    max_steps: int = -1,
) -> dict:
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

    data_dir, out_dir = Path(data_dir), Path(out_dir)
    tok = AutoTokenizer.from_pretrained(base)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    train_rows = [
        encode(tok, r_["messages"], max_len) for r_ in read_jsonl(data_dir / "train.jsonl")
    ]
    val_path = data_dir / "val.jsonl"
    val_rows = (
        [encode(tok, r_["messages"], max_len) for r_ in read_jsonl(val_path)]
        if val_path.exists()
        else []
    )

    cuda = torch.cuda.is_available()
    base_model = AutoModelForCausalLM.from_pretrained(
        base, dtype=torch.bfloat16 if cuda else torch.float32
    )
    model = get_peft_model(
        base_model,
        LoraConfig(
            r=r,
            lora_alpha=alpha,
            lora_dropout=0.05,
            target_modules="all-linear",
            task_type="CAUSAL_LM",
        ),
    )
    # transformers 5 folds warmup_ratio into a float warmup_steps; 4.x still has warmup_ratio
    params = inspect.signature(TrainingArguments.__init__).parameters
    warmup: dict[str, Any] = (
        {"warmup_ratio": 0.03} if "warmup_ratio" in params else {"warmup_steps": 0.03}
    )
    args = TrainingArguments(
        output_dir=str(out_dir / "checkpoints"),
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        num_train_epochs=epochs,
        max_steps=max_steps,
        learning_rate=lr,
        lr_scheduler_type="cosine",
        **warmup,
        logging_steps=10,
        eval_strategy="epoch" if val_rows and max_steps < 0 else "no",
        save_strategy="no",
        bf16=cuda,
        report_to=[],
        remove_unused_columns=False,
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_rows,
        eval_dataset=val_rows or None,
        data_collator=_Collator(tok.pad_token_id),
    )
    result = trainer.train()
    model.save_pretrained(str(out_dir))
    tok.save_pretrained(out_dir)
    meta = {
        "base": base,
        "n_train": len(train_rows),
        "n_val": len(val_rows),
        "train_loss": float(result.training_loss),
        "lora_r": r,
        "epochs": epochs,
        "lr": lr,
    }
    if val_rows and max_steps < 0:
        meta["eval_loss"] = float(trainer.evaluate()["eval_loss"])
    (out_dir / "train_meta.json").write_text(json.dumps(meta, indent=2))
    return meta


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--base", required=True, help="HF id or local path of the student base model")
    p.add_argument("--data", required=True, help="directory with train.jsonl / val.jsonl")
    p.add_argument("--out", required=True, help="where to write the LoRA adapter")
    p.add_argument("--r", type=int, default=16)
    p.add_argument("--alpha", type=int, default=32)
    p.add_argument("--epochs", type=float, default=2.0)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--max-len", type=int, default=2048)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--max-steps", type=int, default=-1)
    a = p.parse_args(argv)
    meta = train(
        a.base,
        a.data,
        a.out,
        a.r,
        a.alpha,
        a.epochs,
        a.lr,
        a.max_len,
        a.batch_size,
        a.grad_accum,
        a.max_steps,
    )
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
