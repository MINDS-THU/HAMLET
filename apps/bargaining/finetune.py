#!/usr/bin/env python
"""
gemma_finetune.py
~~~~~~~~~~~~~~~~~
Pipeline to

  • ingest Langfuse-exported conversations under  finetuning_dataset/
  • convert them with Gemma's chat template (+generation prompt)
  • fine-tune google/gemma-3-27b-it with 4-bit QLoRA via TRL-SFTTrainer

Assumes an A100-80 GB (or two A100-40 GB) node.
"""

# ---------- imports ----------
import os
import json
from typing import List, Dict

import torch
from datasets import Dataset, load_from_disk

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
)

from trl import SFTTrainer                              # TRL trainer
from peft import LoraConfig                             # PEFT LoRA

# ---------- user paths ----------
DATA_ROOT   = "finetuning_dataset"              # session folders live here
DS_DISK_DIR = "gemma_chat_dataset"              # processed HF-dataset path
MODEL_NAME  = "google/gemma-3-27b-it"           # 27-B instruct checkpoint
CHECKPOINT_BASE = "/gpfs/radev/scratch/zhuoran_yang/cl2637"
OUT_DIR     = os.path.join(CHECKPOINT_BASE, "gemma-27b-qlora-finetuned")

# ======================================================
# 1)  DATA INGEST  +  CONVERSION  WITH CHAT TEMPLATE
# ======================================================
def load_langfuse_jsons(root: str) -> List[Dict]:
    """Walk `root` and turn every prompt/completion pair into a single chat list."""
    samples = []
    # for session in os.listdir(root):
    for session in ['20250515_173736_amazon_499_bf_1_2_sf_0_8', '20250515_173836_amazon_499_bf_0_8_sf_1_2']:
        session_path = os.path.join(root, session)
        if not os.path.isdir(session_path):
            continue
        for fname in os.listdir(session_path):
            if fname.endswith(".json"):
                with open(os.path.join(session_path, fname)) as f:
                    records = json.load(f)
                for rec in records:
                    turns = rec["prompt"] + [rec["completion"]]
                    samples.append({"chat": turns})
    return samples

# ------------------------------------------------------
# helper: merge consecutive same‑role messages and enforce alternation
# ------------------------------------------------------
def _canonicalize_chat(chat: List[Dict]) -> List[Dict]:
    """
    • merge consecutive messages that share the same role (A + '\n' + B)
    • allow roles: system / user / assistant
    • after an optional system, keep strict user ↔ assistant alternation
    • ensure the last visible role is *user* so that add_generation_prompt works
    """
    allowed = {"system", "user", "assistant"}
    merged  = []

    # (A) merge duplicates and drop unknown roles
    for msg in chat:
        role = msg["role"]
        if role not in allowed:
            continue
        if merged and merged[-1]["role"] == role:
            merged[-1]["content"] += "\n" + msg["content"]
        else:
            merged.append({"role": role, "content": msg["content"]})

    if not merged:
        return []

    # (B) keep at most one leading system
    if merged[0]["role"] == "system":
        system_msg = merged[0]
        rest       = merged[1:]
    else:
        system_msg = None
        rest       = merged

    # (C) enforce user/assistant alternation
    alternated, expect = [], "user"
    for msg in rest:
        if msg["role"] != expect:
            continue
        alternated.append(msg)
        expect = "assistant" if expect == "user" else "user"

    if system_msg:
        alternated.insert(0, system_msg)

    # (D) final role must be user
    if alternated and alternated[-1]["role"] == "assistant":
        alternated.pop()

    return alternated


def build_chat_dataset(data_root: str, tokenizer) -> Dataset:
    """Apply Gemma chat template and output a single text column."""
    raw = load_langfuse_jsons(data_root)
    hf_ds = Dataset.from_list(raw).shuffle(seed=42)     # one-time shuffle

    def _templater(ex):
        safe_chat = _canonicalize_chat(ex["chat"])
        return {
            "text": tokenizer.apply_chat_template(
                safe_chat,
                tokenize=False,
                add_generation_prompt=True   # appends assistant header
            )
        }

    hf_ds = hf_ds.map(_templater, remove_columns=["chat"])
    return hf_ds


def prepare_dataset():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token          # Gemma uses EOS as pad
    dataset = build_chat_dataset(DATA_ROOT, tokenizer)
    dataset.save_to_disk(DS_DISK_DIR)
    print(f"✅ Saved processed dataset to {DS_DISK_DIR} with {len(dataset):,} samples.")


# ======================================================
# 2)  QLoRA CONFIG + SFT TRAINING
# ======================================================
def finetune():
    os.makedirs(OUT_DIR, exist_ok=True)                # ensure scratch dir exists

    # ---- load dataset & tokenizer ----
    dataset   = load_from_disk(DS_DISK_DIR)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    # ---- 4-bit quantisation ----
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,         # bf16 on A100
    )

    # ---- LoRA (QLoRA) ----
    peft_cfg = LoraConfig(
        r=32,
        lora_alpha=64,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # ---- model ----
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_cfg,
        device_map="auto",
        trust_remote_code=True,
    )

    # ---- training args ----
    args = TrainingArguments(
        output_dir          = OUT_DIR,
        num_train_epochs    = 3,
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 16,   # effective batch = 32
        learning_rate       = 2e-5,
        lr_scheduler_type   = "cosine",
        warmup_ratio        = 0.03,
        logging_steps       = 25,
        save_strategy       = "epoch",
        bf16                = True,         # works on A100
        report_to           = "none",
    )

    # ---- trainer ----
    trainer = SFTTrainer(
        model              = model,
        tokenizer          = tokenizer,
        train_dataset      = dataset,
        peft_config        = peft_cfg,
        dataset_text_field = "text",
        args               = args,
    )

    trainer.train()
    trainer.save_model(OUT_DIR)
    print("🎉 Fine-tuning complete. Checkpoints written to:", OUT_DIR)


# -------------------- CLI --------------------
if __name__ == "__main__":
    import argparse, warnings
    warnings.filterwarnings("ignore", category=UserWarning)

    parser = argparse.ArgumentParser(
        description="Gemma-27B-IT QLoRA finetuning pipeline")
    parser.add_argument(
        "--stage", choices=["prepare", "train", "all"], default="all",
        help="prepare = build dataset only; train = expect pre-built dataset; all = both")
    args = parser.parse_args()

    if args.stage in ("prepare", "all"):
        prepare_dataset()
    if args.stage in ("train", "all"):
        finetune()
