"""T6, distillation: a big teacher LLM labels a few thousand articles; a small
open model learns to copy it with LoRA, then runs on everything, fast and free.

dataset      stratified sample (training period only) -> teacher labels -> chat JSONL
train_lora   LoRA supervised fine-tuning (GPU server; verified on a tiny model in tests)
evaluate     field-by-field agreement between teacher and student extractions

Workflow on the GPU server:
    newsalpha distill-data --config configs/distill.yaml          # ~$30-40 of teacher API
    python -m newsalpha.distill.train_lora --base <model> --data <dir> --out <adapter>
    vllm serve <model> --enable-lora --lora-modules student=<adapter>
    newsalpha distill-eval --config configs/distill.yaml          # student vs teacher on val
"""
