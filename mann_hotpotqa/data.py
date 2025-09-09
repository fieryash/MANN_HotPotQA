import re
from typing import List, Dict, Tuple, Optional

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

try:
    from rank_bm25 import BM25Okapi
except Exception:
    BM25Okapi = None  # optional

try:
    from bert_score import score as bert_score
except Exception:
    bert_score = None  # optional

from transformers import AutoTokenizer


def retrieve_top_k_bm25(question: str, context_dict: Dict, k: int = 2) -> List[str]:
    """Return top-k paragraph strings using BM25 over provided context dict.

    Expects Hotpot-style context: {"title": [...], "sentences": [[...], ...]}
    """
    if BM25Okapi is None:
        # Fallback: simple lexical scoring
        paragraphs = []
        for sents in context_dict.get("sentences", []):
            para = " ".join(sents)
            paragraphs.append(para)
        # Rank by term overlap as naive fallback
        q_tokens = set(question.lower().split())
        scored = []
        for p in paragraphs:
            p_tokens = set(p.lower().split())
            scored.append((len(q_tokens & p_tokens), p))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [p for _, p in scored[:k]]

    paragraphs = []
    para_texts = []
    for sents in context_dict.get("sentences", []):
        para = " ".join(sents)
        paragraphs.append(para)
        para_texts.append(para)

    tokenized = [p.lower().split() for p in para_texts]
    bm25 = BM25Okapi(tokenized)
    scores = bm25.get_scores(question.lower().split())
    top_indices = sorted(range(len(scores)), key=lambda i: float(scores[i]), reverse=True)[:k]
    return [paragraphs[i] for i in top_indices]


def retrieve_top_k_bertscore(question: str, context_dict: Dict, k: int = 2, model_type: str = "bert-base-uncased") -> List[str]:
    """Optional: Rank paragraphs by BERTScore F1."""
    if bert_score is None:
        return retrieve_top_k_bm25(question, context_dict, k=k)

    paragraphs = []
    paragraph_texts = []
    for sents in context_dict.get("sentences", []):
        para = " ".join(sents)
        paragraphs.append(para)
        paragraph_texts.append(para)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    _, _, f1 = bert_score(paragraph_texts, [question] * len(paragraph_texts), model_type=model_type, verbose=False, device=device)
    scores = f1.tolist()
    top_indices = sorted(range(len(scores)), key=lambda i: float(scores[i]), reverse=True)[:k]
    return [paragraphs[i] for i in top_indices]


def tokenize_segmented(example: Dict, tokenizer: AutoTokenizer, max_length: int = 384, k: int = 2, retrieval: str = "bm25") -> Dict:
    """Tokenize question + concatenated top-k paragraphs, return span indices if answer found.

    example: {"question": str, "context": {"title": [...], "sentences": [[...], ...]}, "answer": str}
    """
    question = example.get("question", "")
    context_dict = example.get("context", {})
    answer = (example.get("answer") or "").strip()

    if retrieval == "bertscore":
        top_paragraphs = retrieve_top_k_bertscore(question, context_dict, k=k)
    else:
        top_paragraphs = retrieve_top_k_bm25(question, context_dict, k=k)

    segmented_context = " [SEP] ".join(top_paragraphs)

    encoded = tokenizer(
        question,
        segmented_context,
        truncation=True,
        max_length=max_length,
        return_offsets_mapping=True,
        return_token_type_ids=True,
    )

    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]
    token_type_ids = encoded.get("token_type_ids") or [0] * len(input_ids)
    offsets = encoded["offset_mapping"]

    context_token_indices = [i for i, ttid in enumerate(token_type_ids) if ttid == 1]

    if not answer:
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "start_index": -1,
            "end_index": -1,
        }

    answer_norm = re.sub(r"\s+", " ", answer.lower())
    context_norm = segmented_context.lower()
    match = re.search(re.escape(answer_norm), context_norm)
    if not match:
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "start_index": -1,
            "end_index": -1,
        }

    start_char, end_char = match.start(), match.end()
    start_idx = -1
    end_idx = -1
    for idx in context_token_indices:
        sub_start, sub_end = offsets[idx]
        if start_idx == -1 and sub_start >= start_char:
            start_idx = idx
        if sub_end <= end_char:
            end_idx = idx

    if start_idx == -1 or end_idx == -1 or end_idx < start_idx:
        start_idx, end_idx = -1, -1

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "start_index": start_idx,
        "end_index": end_idx,
    }


class HotpotPreprocessedDataset(Dataset):
    """Thin wrapper over a HuggingFace dataset already containing tokenized items."""

    def __init__(self, hf_dataset):
        self.data = hf_dataset

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return (
            torch.LongTensor(item["input_ids"]),
            torch.LongTensor(item["attention_mask"]),
            item["start_index"],
            item["end_index"],
        )


def collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor, int, int]]):
    input_ids_list = [x[0] for x in batch]
    attn_list = [x[1] for x in batch]
    starts = torch.LongTensor([x[2] for x in batch])
    ends = torch.LongTensor([x[3] for x in batch])

    input_ids_padded = pad_sequence(input_ids_list, batch_first=True, padding_value=0)
    attn_padded = pad_sequence(attn_list, batch_first=True, padding_value=0)

    return input_ids_padded, attn_padded, starts, ends


def get_tokenizer(name: str = "bert-base-uncased") -> AutoTokenizer:
    return AutoTokenizer.from_pretrained(name, use_fast=True)

