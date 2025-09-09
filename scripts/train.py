import argparse
import os
import time

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from datasets import load_dataset, load_from_disk

from mann_hotpotqa.data import (
    get_tokenizer,
    tokenize_segmented,
    HotpotPreprocessedDataset,
    collate_fn,
)
from mann_hotpotqa.model import AdvancedMANN_QA
from mann_hotpotqa.train_utils import train_one_epoch, evaluate
from mann_hotpotqa.inference import save_checkpoint


def parse_args():
    p = argparse.ArgumentParser(description="Train Advanced MANN QA on HotpotQA")
    p.add_argument("--cache_dir", type=str, default="./cache", help="Cache directory for processed datasets")
    p.add_argument("--train_cache", type=str, default="./cache/hotpot_train")
    p.add_argument("--val_cache", type=str, default="./cache/hotpot_val")
    p.add_argument("--base_model", type=str, default="bert-base-uncased")
    p.add_argument("--max_length", type=int, default=384)
    p.add_argument("--top_k", type=int, default=2)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--hidden_dim", type=int, default=128)
    p.add_argument("--memory_size", type=int, default=32)
    p.add_argument("--memory_dim", type=int, default=64)
    p.add_argument("--freeze_bert_layers", type=int, default=2)
    p.add_argument("--lr_bert", type=float, default=3e-5)
    p.add_argument("--lr_other", type=float, default=1e-3)
    p.add_argument("--warmup_ratio", type=float, default=0.1)
    p.add_argument("--checkpoint", type=str, default="./checkpoints/mann_hotpotqa.ckpt")
    p.add_argument("--num_proc", type=int, default=4)
    p.add_argument("--retrieval", type=str, choices=["bm25", "bertscore"], default="bm25")
    return p.parse_args()


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def main():
    args = parse_args()
    ensure_dir(os.path.dirname(args.train_cache))
    ensure_dir(os.path.dirname(args.checkpoint))

    tokenizer = get_tokenizer(args.base_model)

    # Prepare dataset caches
    if os.path.exists(args.train_cache) and os.path.exists(args.val_cache):
        train_hf = load_from_disk(args.train_cache)
        val_hf = load_from_disk(args.val_cache)
    else:
        hotpot = load_dataset("hotpot_qa", "distractor", trust_remote_code=True)
        train_hf = hotpot["train"].map(
            tokenize_segmented,
            fn_kwargs={
                "tokenizer": tokenizer,
                "max_length": args.max_length,
                "k": args.top_k,
                "retrieval": args.retrieval,
            },
            remove_columns=hotpot["train"].column_names,
            num_proc=args.num_proc,
        )
        val_hf = hotpot["validation"].map(
            tokenize_segmented,
            fn_kwargs={
                "tokenizer": tokenizer,
                "max_length": args.max_length,
                "k": args.top_k,
                "retrieval": args.retrieval,
            },
            remove_columns=hotpot["validation"].column_names,
            num_proc=args.num_proc,
        )
        train_hf.save_to_disk(args.train_cache)
        val_hf.save_to_disk(args.val_cache)

    train_ds = HotpotPreprocessedDataset(train_hf)
    val_ds = HotpotPreprocessedDataset(val_hf)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, pin_memory=True)

    model = AdvancedMANN_QA(
        hidden_dim=args.hidden_dim,
        memory_size=args.memory_size,
        memory_dim=args.memory_dim,
        base_model=args.base_model,
        freeze_bert_layers=args.freeze_bert_layers,
    )
    if torch.cuda.is_available():
        model = model.cuda()
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    bert_params = [p for n, p in model.named_parameters() if "bert" in n and p.requires_grad]
    other_params = [p for n, p in model.named_parameters() if "bert" not in n]
    optimizer = optim.AdamW([
        {"params": bert_params, "lr": args.lr_bert},
        {"params": other_params, "lr": args.lr_other},
    ])

    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(args.warmup_ratio * total_steps),
        num_training_steps=total_steps,
    )

    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}")
        start = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, scheduler, scaler)
        val_loss, val_acc = evaluate(model, val_loader)
        print(f"Epoch {epoch} | Train loss {train_loss:.4f} acc {train_acc:.4f} | Val loss {val_loss:.4f} acc {val_acc:.4f} | {time.time()-start:.1f}s")

    # Save checkpoint
    config = {
        "base_model": args.base_model,
        "hidden_dim": args.hidden_dim,
        "memory_size": args.memory_size,
        "memory_dim": args.memory_dim,
        "freeze_bert_layers": args.freeze_bert_layers,
        "max_length": args.max_length,
        "top_k": args.top_k,
    }
    # If DataParallel, unwrap
    model_to_save = model.module if hasattr(model, "module") else model
    save_checkpoint(model_to_save, tokenizer, args.checkpoint, config)
    print(f"Saved checkpoint to {args.checkpoint}")


if __name__ == "__main__":
    main()

