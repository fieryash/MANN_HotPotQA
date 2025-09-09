from typing import Dict, List, Tuple, Optional

import json
import torch
from transformers import AutoTokenizer

from .model import AdvancedMANN_QA
from .data import retrieve_top_k_bm25, retrieve_top_k_bertscore


def prepare_input_pair(tokenizer: AutoTokenizer, question: str, context_text: str, max_length: int = 384) -> Tuple[torch.Tensor, torch.Tensor]:
    encoded = tokenizer(
        question,
        context_text,
        truncation=True,
        max_length=max_length,
        return_offsets_mapping=False,
        return_token_type_ids=True,
    )
    input_ids = torch.tensor(encoded["input_ids"]).unsqueeze(0)
    attention_mask = torch.tensor(encoded["attention_mask"]).unsqueeze(0)
    return input_ids, attention_mask


def decode_answer(tokenizer: AutoTokenizer, input_ids: torch.Tensor, start_idx: int, end_idx: int) -> str:
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0].cpu().tolist())
    if start_idx > end_idx or end_idx >= len(tokens) or start_idx < 0:
        return "[Invalid prediction]"
    return tokenizer.convert_tokens_to_string(tokens[start_idx : end_idx + 1])


@torch.no_grad()
def ask_question(model: AdvancedMANN_QA, tokenizer: AutoTokenizer, question: str, context_paragraphs: List[str], max_length: int = 384, k: Optional[int] = None) -> Tuple[str, Tuple[int, int]]:
    # If k provided, pick top-k via BM25 from the provided paragraphs
    if k is not None and len(context_paragraphs) > k:
        # Build a fake hotpot-style dict for reuse
        context_dict = {"sentences": [[p] for p in context_paragraphs]}
        top_paragraphs = retrieve_top_k_bm25(question, context_dict, k=k)
    else:
        top_paragraphs = context_paragraphs

    context_text = " [SEP] ".join(top_paragraphs)
    input_ids, attention_mask = prepare_input_pair(tokenizer, question, context_text, max_length=max_length)
    if torch.cuda.is_available():
        input_ids = input_ids.cuda()
        attention_mask = attention_mask.cuda()
        model = model.cuda()
    model.eval()
    start_logits, end_logits = model(input_ids, attention_mask)
    start_idx = start_logits.argmax(dim=-1).item()
    end_idx = end_logits.argmax(dim=-1).item()
    answer = decode_answer(tokenizer, input_ids, start_idx, end_idx)
    return answer, (start_idx, end_idx)


def save_checkpoint(model: AdvancedMANN_QA, tokenizer: AutoTokenizer, path: str, config: Dict):
    to_save = {
        "state_dict": model.state_dict(),
        "config": config,
        "tokenizer": tokenizer.name_or_path,
    }
    torch.save(to_save, path)


def load_model(checkpoint_path: str) -> Tuple[AdvancedMANN_QA, AutoTokenizer, Dict]:
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    cfg = ckpt.get("config", {})
    base_model = cfg.get("base_model", "bert-base-uncased")
    hidden_dim = int(cfg.get("hidden_dim", 128))
    memory_size = int(cfg.get("memory_size", 32))
    memory_dim = int(cfg.get("memory_dim", 64))
    freeze_bert_layers = int(cfg.get("freeze_bert_layers", 2))

    model = AdvancedMANN_QA(hidden_dim=hidden_dim, memory_size=memory_size, memory_dim=memory_dim, base_model=base_model, freeze_bert_layers=freeze_bert_layers)
    model.load_state_dict(ckpt["state_dict"], strict=False)
    tokenizer = AutoTokenizer.from_pretrained(ckpt.get("tokenizer", base_model))
    return model, tokenizer, cfg

