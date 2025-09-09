from typing import Tuple, Optional

import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm


def qa_span_loss(start_logits: torch.Tensor, end_logits: torch.Tensor, start_positions: torch.Tensor, end_positions: torch.Tensor) -> torch.Tensor:
    bsz, _ = start_logits.size()
    valid_mask = (start_positions >= 0) & (end_positions >= 0)
    if valid_mask.sum() == 0:
        return torch.tensor(0.0, requires_grad=True, device=start_logits.device)

    valid_starts = start_positions[valid_mask]
    valid_ends = end_positions[valid_mask]
    valid_start_logits = start_logits[valid_mask]
    valid_end_logits = end_logits[valid_mask]

    loss_fct = nn.CrossEntropyLoss()
    loss_start = loss_fct(valid_start_logits, valid_starts)
    loss_end = loss_fct(valid_end_logits, valid_ends)
    return loss_start + loss_end


def train_one_epoch(model, loader, optimizer, scheduler: Optional[object], scaler: GradScaler, desc: str = "Training") -> Tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_count = 0
    total_correct = 0

    for batch in tqdm(loader, desc=desc, leave=False):
        input_ids, attn_mask, starts, ends = batch
        input_ids = input_ids.cuda() if torch.cuda.is_available() else input_ids
        attn_mask = attn_mask.cuda() if torch.cuda.is_available() else attn_mask
        starts = starts.cuda() if torch.cuda.is_available() else starts
        ends = ends.cuda() if torch.cuda.is_available() else ends

        optimizer.zero_grad()

        with autocast(enabled=torch.cuda.is_available()):
            start_logits, end_logits = model(input_ids, attn_mask)
            loss = qa_span_loss(start_logits, end_logits, starts, ends)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        if scheduler is not None:
            scheduler.step()

        valid_mask = (starts >= 0)
        pred_start = start_logits.argmax(dim=-1)
        pred_end = end_logits.argmax(dim=-1)
        correct = ((pred_start == starts) & (pred_end == ends) & valid_mask).sum().item()
        total_valid = valid_mask.sum().item()

        count = max(total_valid, 1)
        total_loss += loss.item() * count
        total_count += count
        total_correct += correct

    avg_loss = total_loss / max(total_count, 1)
    avg_acc = total_correct / max(total_count, 1)
    return avg_loss, avg_acc


def evaluate(model, loader, desc: str = "Validating") -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_count = 0
    correct_spans = 0
    total_spans = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc=desc, leave=False):
            input_ids, attn_mask, starts, ends = batch
            input_ids = input_ids.cuda() if torch.cuda.is_available() else input_ids
            attn_mask = attn_mask.cuda() if torch.cuda.is_available() else attn_mask
            starts = starts.cuda() if torch.cuda.is_available() else starts
            ends = ends.cuda() if torch.cuda.is_available() else ends

            start_logits, end_logits = model(input_ids, attn_mask)
            loss = qa_span_loss(start_logits, end_logits, starts, ends)

            count = (starts >= 0).sum().item()
            total_loss += loss.item() * max(count, 1)
            total_count += max(count, 1)

            valid_mask = (starts >= 0) & (ends >= 0)
            pred_start = start_logits.argmax(dim=-1)
            pred_end = end_logits.argmax(dim=-1)

            valid_indices = torch.where(valid_mask)[0]
            for i in valid_indices:
                total_spans += 1
                if pred_start[i] == starts[i] and pred_end[i] == ends[i]:
                    correct_spans += 1

    avg_loss = total_loss / max(total_count, 1)
    span_acc = correct_spans / max(total_spans, 1)
    return avg_loss, span_acc

