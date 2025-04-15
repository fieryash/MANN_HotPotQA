
# 🔍 Advanced Memory-Augmented QA Model for HotpotQA

This project implements a hybrid **Memory-Augmented Neural Network (MANN)** for **multi-hop question answering** on the HotpotQA dataset using a combination of:

- ✅ BERT Embeddings
- 🧠 Neural Turing Machine (NTM) memory module
- 🔁 LSTM-based reasoning
- 🔎 BM25 paragraph retrieval

---

## 🧠 Key Highlights

- **Dataset**: HotpotQA (distractor setting)
- **Retrieval**: BM25-based top-k paragraph selection
- **Architecture**: BERT + LSTM + NTM Memory
- **Training**: Mixed precision (AMP), span-level supervision
- **Evaluation**: Exact Match (EM) and F1 Score
- **Manual QA**: Ask questions with custom context

---

## 📦 Installation

```bash
pip install datasets bert-score rank_bm25 transformers torch tqdm
```

---

## 📚 Dataset: HotpotQA

We use the `hotpot_qa` dataset with `distractor` configuration:

```python
from datasets import load_dataset
hotpot = load_dataset("hotpot_qa", "distractor")
```

Each sample includes:
- `question`, `answer`
- `context`: 10 paragraphs
- `supporting_facts`: for multi-hop reasoning

---

## 🔎 Retrieval: BM25

Fast, unsupervised relevance scoring:

```python
def retrieve_top_k_bm25(question, context_dict, k=2):
    # Ranks paragraphs based on keyword overlap
    ...
```

> BERTScore retrieval is implemented but commented out due to GPU cost.

---

## ⚙️ Preprocessing

- Top-k paragraphs joined with `[SEP]`
- BERT tokenizer with max length = 384
- Maps answer span from text to token indices
- Cached to disk via Hugging Face Datasets

```python
train_proc.save_to_disk("hotpot_cached_train/")
val_proc.save_to_disk("hotpot_cached_val/")
```

---

## 🧱 Model: `AdvancedMANN_QA`

| Component      | Description                                  |
|----------------|----------------------------------------------|
| BERT           | Token embeddings                             |
| LSTM Layer 1   | Sequential token encoding                    |
| NTM Memory     | Differentiable read/write memory             |
| LSTM Layer 2   | Memory-augmented sequence reasoning          |
| QA Head        | Predicts start and end span logits           |

---

## 🏋️ Training

- Mixed precision (`torch.amp`)
- Optimizer: AdamW
- Scheduler: Warmup + linear decay
- Two-tier LR:
  - BERT: `3e-5`
  - Other layers: `1e-3`

```python
loss = qa_span_loss(start_logits, end_logits, start_idx, end_idx)
```

---

## ✅ Evaluation

### Metrics:
- **Exact Match (EM)**: Normalized answer match
- **F1 Score**: Token-level overlap

### Example Results:
```
EM: 62.5%
F1: 90.5%
```

---

## 🔍 Inference Example

```python
question = "Who owns Radio City FM?"
context = [
    ["Radio City", [
        "Radio City is India's first private FM radio station.",
        "Radio City was acquired by Music Broadcast Ltd."
    ]]
]

answer, span = ask_question(model, tokenizer, question, context)
print("Answer:", answer)
```

---

## 💾 Saving / Loading

```python
# Save
torch.save(model.module.state_dict(), "mann_qa_model.pt")

# Load
model.load_state_dict(torch.load("mann_qa_model.pt"))
```

---

## 📝 References

- [HotpotQA Dataset](https://huggingface.co/datasets/hotpot_qa)
- [Neural Turing Machines](https://arxiv.org/abs/1410.5401)
- [BERTScore](https://github.com/Tiiiger/bert_score)
- [Transformers](https://github.com/huggingface/transformers)

---
