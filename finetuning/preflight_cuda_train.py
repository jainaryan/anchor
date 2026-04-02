#!/usr/bin/env python3
"""
Fast preflight checks for CUDA QLoRA training.

This script validates:
1) cleaned JSONL files exist and have the expected schema
2) assistant-only masking yields labels aligned with input_ids
3) batch collation pads input_ids and labels to compatible shapes

It intentionally does NOT load model weights, so it is cheap enough to run
on a login node before submitting a GPU job.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List

from transformers import AutoTokenizer, DataCollatorForSeq2Seq


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="meta-llama/Llama-3.2-3B-Instruct",
        help="Tokenizer source (HF repo id or local path).",
    )
    parser.add_argument(
        "--train-file",
        default="data/conversations_cleaned/mindmate_train.jsonl",
        help="Path to cleaned train JSONL.",
    )
    parser.add_argument(
        "--val-file",
        default="data/conversations_cleaned/mindmate_val.jsonl",
        help="Path to cleaned validation JSONL.",
    )
    parser.add_argument(
        "--samples-per-split",
        type=int,
        default=128,
        help="How many examples to validate from each split.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=2048,
        help="Tokenization truncation length.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Synthetic collation batch size for validation.",
    )
    return parser.parse_args()


def load_jsonl_examples(path: Path, limit: int) -> List[Dict]:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    out: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= limit:
                break
            obj = json.loads(line)
            conv = obj.get("conversations")
            if not isinstance(conv, list) or not conv:
                raise ValueError(f"{path} line {i + 1}: missing/empty conversations list")

            roles = [m.get("role") for m in conv if isinstance(m, dict)]
            if "assistant" not in roles:
                raise ValueError(f"{path} line {i + 1}: no assistant turn present")
            if "user" not in roles:
                raise ValueError(f"{path} line {i + 1}: no user turn present")

            out.append({"conversations": conv})

    if not out:
        raise ValueError(f"{path}: no examples loaded")
    return out


def to_feature(example: Dict, tokenizer, max_length: int) -> Dict[str, List[int]]:
    header_token = "<|start_header_id|>assistant<|end_header_id|>\n\n"
    eot_token = "<|eot_id|>"

    text = tokenizer.apply_chat_template(
        example["conversations"],
        tokenize=False,
        add_generation_prompt=False,
    )
    tok = tokenizer(
        text,
        truncation=True,
        max_length=max_length,
        padding=False,
        return_offsets_mapping=True,
        add_special_tokens=False,
    )

    input_ids = tok["input_ids"]
    attention_mask = tok["attention_mask"]
    offsets = tok["offset_mapping"]
    labels = [-100] * len(input_ids)

    search_start = 0
    while True:
        start_idx = text.find(header_token, search_start)
        if start_idx == -1:
            break

        content_start = start_idx + len(header_token)
        end_idx = text.find(eot_token, content_start)
        content_end = len(text) if end_idx == -1 else end_idx + len(eot_token)

        for j, (tok_start, tok_end) in enumerate(offsets):
            if tok_start >= content_start and tok_end <= content_end and tok_start < tok_end:
                labels[j] = input_ids[j]

        search_start = content_end
        if end_idx == -1:
            break

    if not (len(input_ids) == len(attention_mask) == len(labels)):
        raise ValueError(
            "Length mismatch in tokenized feature: "
            f"input_ids={len(input_ids)} attention_mask={len(attention_mask)} labels={len(labels)}"
        )

    supervised = sum(1 for x in labels if x != -100)
    if supervised == 0:
        raise ValueError("No assistant-supervised tokens found in sample")

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def validate_split(split_name: str, examples: List[Dict], tokenizer, max_length: int, batch_size: int) -> None:
    features = [to_feature(ex, tokenizer, max_length=max_length) for ex in examples]

    collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=None,
        padding=True,
        pad_to_multiple_of=8,
        label_pad_token_id=-100,
    )

    for start in range(0, len(features), batch_size):
        batch = collator(features[start : start + batch_size])
        if batch["input_ids"].shape != batch["labels"].shape:
            raise ValueError(
                f"{split_name}: collated shape mismatch "
                f"input_ids={tuple(batch['input_ids'].shape)} "
                f"labels={tuple(batch['labels'].shape)}"
            )

    print(f"[ok] {split_name}: {len(features)} samples validated")


def main() -> None:
    args = parse_args()

    train_file = Path(args.train_file)
    val_file = Path(args.val_file)

    print(f"[info] loading tokenizer from: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_examples = load_jsonl_examples(train_file, args.samples_per_split)
    val_examples = load_jsonl_examples(val_file, args.samples_per_split)

    validate_split(
        "train",
        train_examples,
        tokenizer=tokenizer,
        max_length=args.max_length,
        batch_size=args.batch_size,
    )
    validate_split(
        "validation",
        val_examples,
        tokenizer=tokenizer,
        max_length=args.max_length,
        batch_size=args.batch_size,
    )

    print("[success] CUDA training preflight passed")


if __name__ == "__main__":
    main()

