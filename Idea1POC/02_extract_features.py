#!/usr/bin/env python
"""
Step 2 -- Extract frozen LM features for every TR text window.

Produces features/<tag>/layer{L}.npy  with shape (T, d) per requested layer,
mean-pooled over tokens of the window (anchor paper pools the last block;
we save several layers so you can pick the best-aligning one).

Also works for a LoRA-fine-tuned checkpoint (step 4 output) via --adapter.

Usage
-----
  python 02_extract_features.py --prepared data/prepared --model gpt2 \
         --layers 6 9 12 --out features/gpt2_base
  python 02_extract_features.py --prepared data/prepared --model meta-llama/Llama-3.2-1B \
         --adapter runs/misalign_md_preserve_lang/adapter --out features/llama_misMD
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
from transformers import AutoModel, AutoModelForCausalLM, AutoModelForMaskedLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import PreparedDataset  # noqa: E402


def load_model(name: str, adapter: str | None, device: str):
    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token if tok.eos_token else tok.unk_token
    is_mlm = "bert" in name.lower()
    cls = AutoModelForMaskedLM if is_mlm else AutoModelForCausalLM
    model = cls.from_pretrained(name, torch_dtype=torch.float32)
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
        model = model.merge_and_unload()
    model.eval().to(device)
    return tok, model, is_mlm


@torch.no_grad()
def extract(texts, tok, model, layers, device, batch_size=16, max_len=256):
    feats = {l: [] for l in layers}
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        enc = tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=max_len).to(device)
        out = model(**enc, output_hidden_states=True)
        hs = out.hidden_states  # tuple(len = n_layers+1) of (B, L, d)
        mask = enc["attention_mask"].unsqueeze(-1).float()
        for l in layers:
            h = hs[l]
            pooled = (h * mask).sum(1) / mask.sum(1).clamp(min=1)
            feats[l].append(pooled.float().cpu().numpy())
    return {l: np.vstack(v) for l, v in feats.items()}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepared", required=True)
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--layers", type=int, nargs="+", default=None, help="hidden_states indices; default = last")
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch_size", type=int, default=16)
    a = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ds = PreparedDataset.load(a.prepared)
    tok, model, _ = load_model(a.model, a.adapter, device)
    n_layers = model.config.num_hidden_layers
    layers = a.layers or [n_layers]
    print(f"model={a.model} adapter={a.adapter} layers={layers} T={len(ds.text_windows)}")
    feats = extract(ds.text_windows, tok, model, layers, device, a.batch_size)
    os.makedirs(a.out, exist_ok=True)
    for l, F in feats.items():
        np.save(os.path.join(a.out, f"layer{l}.npy"), F)
        print(f"  saved layer{l}: {F.shape}")
