from typing import Tuple

import torch
import torch.nn as nn
from transformers import AutoModel


class NTM_Memory(nn.Module):
    def __init__(self, memory_size: int, memory_dim: int, num_read_heads: int = 1):
        super().__init__()
        self.memory_size = memory_size
        self.memory_dim = memory_dim
        self.num_read_heads = num_read_heads

    def content_address(self, memory: torch.Tensor, key: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
        mem_norm = nn.functional.normalize(memory, dim=-1)
        key_norm = nn.functional.normalize(key, dim=-1)
        sim = torch.bmm(mem_norm, key_norm.unsqueeze(2)).squeeze(-1)
        sim = beta.unsqueeze(1) * sim
        w = torch.softmax(sim, dim=-1)
        return w

    def write_to_memory(self, memory: torch.Tensor, w: torch.Tensor, e: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        e = torch.sigmoid(e).unsqueeze(1)
        a = a.unsqueeze(1)
        w = w.unsqueeze(-1)
        memory_erase = memory * (1 - w * e)
        memory_add = w * a
        new_mem = memory_erase + memory_add
        return new_mem

    def forward(self, memory: torch.Tensor, interface: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        bsz = memory.size(0)
        dim = memory.size(2)

        offset = 0
        write_key = interface[:, offset : offset + dim]
        offset += dim
        write_beta = torch.relu(interface[:, offset : offset + 1]).squeeze(-1)
        offset += 1
        erase_vec = interface[:, offset : offset + dim]
        offset += dim
        add_vec = interface[:, offset : offset + dim]
        offset += dim
        read_key = interface[:, offset : offset + dim]
        offset += dim
        read_beta = torch.relu(interface[:, offset : offset + 1]).squeeze(-1)

        w_write = self.content_address(memory, write_key, write_beta)
        memory = self.write_to_memory(memory, w_write, erase_vec, add_vec)

        w_read = self.content_address(memory, read_key, read_beta)
        read_vec = torch.bmm(w_read.unsqueeze(1), memory).squeeze(1)  # (b, dim)

        return memory, read_vec


class AdvancedMANN_QA(nn.Module):
    def __init__(self, hidden_dim: int, memory_size: int, memory_dim: int, base_model: str = "bert-base-uncased", freeze_bert_layers: int = 2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.memory_size = memory_size
        self.memory_dim = memory_dim

        self.bert = AutoModel.from_pretrained(base_model)
        if freeze_bert_layers > 0:
            for name, param in self.bert.named_parameters():
                # Freeze lower encoder layers as in the notebook (0 and 1)
                for i in range(freeze_bert_layers):
                    if f"encoder.layer.{i}" in name:
                        param.requires_grad = False
                        break

        bert_dim = self.bert.config.hidden_size

        self.lstm1 = nn.LSTMCell(bert_dim, hidden_dim)

        interface_size = (memory_dim + 1) + memory_dim + memory_dim + (memory_dim + 1)
        self.interface_linear = nn.Linear(hidden_dim, interface_size)
        self.memory_module = NTM_Memory(memory_size, memory_dim, num_read_heads=1)

        self.lstm2 = nn.LSTMCell(hidden_dim + memory_dim, hidden_dim)
        self.proj = nn.Linear(hidden_dim, hidden_dim)

        self.qa_head = nn.Linear(hidden_dim, 2)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return start_logits, end_logits with shape (B, L)."""
        bsz, seq_len = input_ids.size()
        device = input_ids.device

        bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        embeddings = bert_out.last_hidden_state  # (B, L, H)

        h1 = torch.zeros(bsz, self.hidden_dim, device=device)
        c1 = torch.zeros(bsz, self.hidden_dim, device=device)
        h2 = torch.zeros(bsz, self.hidden_dim, device=device)
        c2 = torch.zeros(bsz, self.hidden_dim, device=device)

        memory = torch.zeros(bsz, self.memory_size, self.memory_dim, device=device)
        hidden_states = []

        for t in range(seq_len):
            mask_t = attention_mask[:, t].float().unsqueeze(-1)
            emb_t = embeddings[:, t]

            h1, c1 = self.lstm1(emb_t, (h1, c1))
            interface_vec = self.interface_linear(h1)
            memory, read_vec = self.memory_module(memory, interface_vec)

            inp2 = torch.cat([h1, read_vec], dim=-1)
            h2, c2 = self.lstm2(inp2, (h2, c2))

            hidden_states.append(h2 * mask_t)

        hidden_stack = torch.stack(hidden_states, dim=0).transpose(0, 1)
        hidden_proj = self.proj(hidden_stack)

        logits = self.qa_head(hidden_proj)  # (B, L, 2)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)
        return start_logits, end_logits

