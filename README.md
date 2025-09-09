
# 🔍 Advanced Memory-Augmented QA Model for HotpotQA

This project implements a hybrid **Memory-Augmented Neural Network (MANN)** for **multi-hop question answering** on the HotpotQA dataset using a combination of:

- ✅ BERT Embeddings
- 🧠 Neural Turing Machine (NTM) memory module
- 🔁 LSTM-based reasoning
- 🔎 BM25 paragraph retrieval

---

## Structured Code and Streamlit App

This repo now includes a small Python package and a Streamlit UI built from the notebook logic.

Project layout:

```
mann_hotpotqa/
  __init__.py
  data.py          # retrieval, tokenization, dataset, collate
  model.py         # NTM memory and AdvancedMANN_QA
  train_utils.py   # loss, train/eval loops (AMP)
  inference.py     # load/save checkpoints, ask_question
scripts/
  train.py         # CLI to preprocess+train+save checkpoint
app.py             # Streamlit UI for inference
requirements.txt
Dockerfile
```

### Setup

```
python -m venv .venv
source .venv/bin/activate  # (Windows: .venv\Scripts\Activate)
pip install -r requirements.txt
```

### Train

```
python scripts/train.py \
  --epochs 3 --batch_size 32 \
  --base_model bert-base-uncased \
  --hidden_dim 128 --memory_size 32 --memory_dim 64 \
  --train_cache ./cache/hotpot_train --val_cache ./cache/hotpot_val \
  --checkpoint ./checkpoints/mann_hotpotqa.ckpt
```

This will download HotpotQA (distractor), preprocess with BM25 top-k, train, and save a checkpoint.

### Run Streamlit UI

```
streamlit run app.py
```

In the sidebar, either load the saved checkpoint (`./checkpoints/mann_hotpotqa.ckpt`) or build a fresh model (not fine-tuned).

---

## Deployment Options

- Docker (local or any VM):
  - Build: `docker build -t mann-hotpotqa .`
  - Run: `docker run -p 8501:8501 mann-hotpotqa`
  - Open: http://localhost:8501

- Streamlit Community Cloud:
  - Push this repo to GitHub
  - Create a new app pointing to `app.py`
  - Set Python version and `requirements.txt`

- Hugging Face Spaces (Streamlit):
  - Create a Space (type: Streamlit)
  - Upload the repo (or point to it)
  - Set `app.py` as entry and include `requirements.txt`

- AWS EC2 / Azure VM / GCP VM:
  - Provision a GPU/CPU VM
  - Install Docker and run the image as above, or
  - Install Python + requirements and run `streamlit run app.py`
  - Use a reverse proxy (Nginx) for HTTPS if public

- GCP Cloud Run / AWS App Runner:
  - Use the provided `Dockerfile`
  - Configure service to expose port `8501`

- FastAPI + Uvicorn (API-only, optional):
  - Wrap model loading and `ask_question` into FastAPI endpoints
  - Containerize and deploy behind your preferred platform (Cloud Run, ECS, etc.)

Model artifact to deploy: `./checkpoints/mann_hotpotqa.ckpt` (contains model weights and config; tokenizer path is stored too).


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
