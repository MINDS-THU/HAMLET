#!/usr/bin/env python
"""
finetune.py
~~~~~~~~~~~~~~~~~
Example script to

  • ingest Langfuse-exported conversations under finetuning_dataset/
  • convert them with Gemma's chat template (+generation prompt)
  • fine-tune google/gemma-3-27b-it with 4-bit QLoRA via TRL-SFTTrainer

Assumes an H100-80GB node.
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
)

from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file

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
def merge_consecutive_messages_with_same_role(messages: List[Dict]) -> List[Dict]:
    """Merge consecutive messages with the same role into a single message."""
    merged = []
    for message in messages:
        if not merged or merged[-1]['role'] != message['role']:
            merged.append(message)
        else:
            merged[-1]['content'] += message['content']
    return merged

def load_langfuse_jsons(root: str) -> List[Dict]:
    """Walk `root` and turn every prompt/completion pair into a single chat list."""
    samples = []
    for session in os.listdir(root):
        session_path = os.path.join(root, session)

        if not os.path.isdir(session_path):
            continue
        for fname in os.listdir(session_path):
            if fname.endswith(".json"):
                longest_sample = []
                longest_sample_len = 0
                assert "buyer" in fname or "seller" in fname
                with open(os.path.join(session_path, fname)) as f:
                    records = json.load(f)
                for rec in records:
                    if len(rec["prompt"])+1 > longest_sample_len:
                        longest_sample = rec["prompt"] + [rec["completion"]]
                        longest_sample_len = len(rec["prompt"])+1

                samples.append({"messages": merge_consecutive_messages_with_same_role(longest_sample)})
    return samples

def build_chat_dataset(data_root: str, tokenizer) -> Dataset:
    """Apply Gemma chat template and output a single text column."""
    raw = load_langfuse_jsons(data_root)
    ds = Dataset.from_list(raw).shuffle(seed=42)     # one-time shuffle
    return ds


def prepare_dataset():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token          # Gemma uses EOS as pad
    dataset = build_chat_dataset(DATA_ROOT, tokenizer)
    dataset.save_to_disk(DS_DISK_DIR)
    print(f"✅ Saved processed dataset to {DS_DISK_DIR} with {len(dataset):,} samples.")


# ======================================================
# 2)  QLoRA CONFIG + SFT TRAINING
# ======================================================
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForImageTextToText, BitsAndBytesConfig
from peft import LoraConfig
from trl import SFTConfig
from trl import SFTTrainer

def finetune():
    os.makedirs(OUT_DIR, exist_ok=True)                # ensure scratch dir exists

    # ---- load dataset & tokenizer ----
    dataset   = load_from_disk(DS_DISK_DIR)
    dataset = dataset.train_test_split(test_size=200/2000)

    # Check if GPU benefits from bfloat16
    if torch.cuda.get_device_capability()[0] >= 8:
        torch_dtype = torch.bfloat16
    else:
        torch_dtype = torch.float16

    # Define model init arguments
    model_kwargs = dict(
        attn_implementation="eager", # Use "flash_attention_2" when running on Ampere or newer GPU
        torch_dtype=torch_dtype, # What torch dtype to use, defaults to auto
        device_map="auto", # Let torch decide how to load the model
    )

    # BitsAndBytesConfig: Enables 4-bit quantization to reduce model size/memory usage
    model_kwargs["quantization_config"] = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_compute_dtype=model_kwargs['torch_dtype'],
        bnb_4bit_quant_storage=model_kwargs['torch_dtype'],
    )

    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, **model_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME) # Load the Instruction Tokenizer to use the official Gemma template

    peft_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.05,
        r=16,
        bias="none",
        target_modules="all-linear",
        task_type="CAUSAL_LM",
        modules_to_save=["lm_head", "embed_tokens"] # make sure to save the lm_head and embed_tokens as you train the special tokens
    )

    args = SFTConfig(
        output_dir=OUT_DIR,         # directory to save and repository id
        max_seq_length=512,                     # max sequence length for model and packing of the dataset
        packing=True,                           # Groups multiple samples in the dataset into a single sequence
        num_train_epochs=3,                     # number of training epochs
        per_device_train_batch_size=1,          # batch size per device during training
        gradient_accumulation_steps=4,          # number of steps before performing a backward/update pass
        gradient_checkpointing=True,            # use gradient checkpointing to save memory
        optim="adamw_torch_fused",              # use fused adamw optimizer
        logging_steps=10,                       # log every 10 steps
        save_strategy="epoch",                  # save checkpoint every epoch
        eval_strategy="epoch",                  # evaluate every epoch
        learning_rate=2e-4,                     # learning rate, based on QLoRA paper
        fp16=True if torch_dtype == torch.float16 else False,   # use float16 precision
        bf16=True if torch_dtype == torch.bfloat16 else False,   # use bfloat16 precision
        max_grad_norm=0.3,                      # max gradient norm based on QLoRA paper
        warmup_ratio=0.03,                      # warmup ratio based on QLoRA paper
        lr_scheduler_type="constant",           # use constant learning rate scheduler
        push_to_hub=False,                       # push model to hub
        report_to="tensorboard",                # report metrics to tensorboard
        dataset_kwargs={
            "add_special_tokens": False, # We template with special tokens
            "append_concat_token": True, # Add EOS token as separator token between examples
        }
    )

    # Create Trainer object
    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        peft_config=peft_config,
        processing_class=tokenizer
    )
    trainer.train()
    trainer.save_model(OUT_DIR)
    print("Fine-tuning complete. Checkpoints written to:", OUT_DIR)

    # free the memory again
    del model
    del trainer
    torch.cuda.empty_cache()

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
