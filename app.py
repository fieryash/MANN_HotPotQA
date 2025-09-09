import json
from typing import List

import streamlit as st
import torch

from mann_hotpotqa.inference import load_model, ask_question
from mann_hotpotqa.model import AdvancedMANN_QA
from mann_hotpotqa.data import get_tokenizer


st.set_page_config(page_title="MANN HotpotQA Demo", page_icon="🤖", layout="wide")
st.title("MANN HotpotQA: Memory-Augmented QA Demo")


@st.cache_resource(show_spinner=False)
def load_from_checkpoint(path: str):
    model, tokenizer, cfg = load_model(path)
    if torch.cuda.is_available():
        model = model.cuda()
    model.eval()
    return model, tokenizer, cfg


@st.cache_resource(show_spinner=False)
def build_fresh_model(base_model: str, hidden_dim: int, memory_size: int, memory_dim: int, freeze_bert_layers: int):
    model = AdvancedMANN_QA(hidden_dim=hidden_dim, memory_size=memory_size, memory_dim=memory_dim, base_model=base_model, freeze_bert_layers=freeze_bert_layers)
    tok = get_tokenizer(base_model)
    if torch.cuda.is_available():
        model = model.cuda()
    model.eval()
    return model, tok


with st.sidebar:
    st.header("Model Setup")
    mode = st.radio("Initialization", ["Load checkpoint", "Fresh (pretrained BERT)"])
    model_obj = None
    tokenizer = None
    cfg = {}

    if mode == "Load checkpoint":
        ckpt_path = st.text_input("Checkpoint path", value="./checkpoints/mann_hotpotqa.ckpt")
        load_btn = st.button("Load Model", use_container_width=True)
        if load_btn and ckpt_path:
            model_obj, tokenizer, cfg = load_from_checkpoint(ckpt_path)
            st.success("Checkpoint loaded")
    else:
        base_model = st.text_input("Base model", value="bert-base-uncased")
        hidden_dim = st.number_input("Hidden dim", min_value=32, max_value=1024, value=128, step=32)
        memory_size = st.number_input("Memory size", min_value=8, max_value=256, value=32, step=8)
        memory_dim = st.number_input("Memory dim", min_value=16, max_value=512, value=64, step=16)
        freeze_layers = st.number_input("Freeze BERT layers", min_value=0, max_value=12, value=2, step=1)
        build_btn = st.button("Build Model", use_container_width=True)
        if build_btn:
            model_obj, tokenizer = build_fresh_model(base_model, int(hidden_dim), int(memory_size), int(memory_dim), int(freeze_layers))
            cfg = {
                "base_model": base_model,
                "hidden_dim": int(hidden_dim),
                "memory_size": int(memory_size),
                "memory_dim": int(memory_dim),
                "freeze_bert_layers": int(freeze_layers),
            }
            st.info("Fresh model built (not fine-tuned)")


st.subheader("Ask a Question")
col_q, col_ctx = st.columns([1, 2])
with col_q:
    question = st.text_input("Question", value="Who owns Radio City FM?")
    max_length = st.slider("Max sequence length", min_value=128, max_value=512, value=384, step=32)
    top_k = st.slider("Top-k paragraphs (BM25)", min_value=1, max_value=5, value=2)
    run = st.button("Predict", type="primary")

with col_ctx:
    st.caption("Enter one paragraph per line; blank lines will be merged.")
    default_ctx = """
Radio City is India's first private FM radio station and was started on 3 July 2001.
It broadcasts on 91.1 megahertz from Mumbai, Bengaluru, Lucknow and New Delhi.
Abraham Thomas is the CEO of the company.
Radio City was acquired by Music Broadcast Ltd.
""".strip()
    raw_ctx = st.text_area("Context paragraphs", height=200, value=default_ctx)


def parse_paragraphs(raw: str) -> List[str]:
    # Split on newlines; merge lines into paragraphs by blank line separation
    lines = [ln.strip() for ln in raw.splitlines()]
    paras = []
    buf = []
    for ln in lines:
        if ln:
            buf.append(ln)
        else:
            if buf:
                paras.append(" ".join(buf))
                buf = []
    if buf:
        paras.append(" ".join(buf))
    if not paras and raw.strip():
        paras = [raw.strip()]
    return paras


if run:
    if model_obj is None or tokenizer is None:
        st.warning("Please load or build a model first from the sidebar.")
    else:
        paragraphs = parse_paragraphs(raw_ctx)
        if not paragraphs:
            st.warning("Please provide at least one context paragraph.")
        else:
            with st.spinner("Predicting..."):
                answer, span = ask_question(model_obj, tokenizer, question, paragraphs, max_length=max_length, k=top_k)
            st.success("Prediction complete")
            st.write(f"Predicted span indices: {span}")
            st.markdown(f"**Answer:** {answer}")

st.divider()
st.caption("Tip: For best results, load a fine-tuned checkpoint from training.")

