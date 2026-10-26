#!/usr/bin/env python
"""
Step 4 -- Region-selective brain (mis)alignment fine-tuning.

Extends Merlin & Toneva (ICLR 2026): LoRA fine-tune an LM with
  * an LM head (keep language modelling intact),
  * a TARGET-network brain head behind a gradient-reversal layer (push
    alignment DOWN), and
  * a PRESERVE-network brain head trained normally (hold alignment UP).

    L = w_lm * L_lm
      + w_target   * L_brain(target_head(GRL(h)),  Y_target)
      + w_preserve * L_brain(preserve_head(h),     Y_preserve)

Conditions (--condition):
  misalign_md_preserve_lang   target=MD,   preserve=LANG     (Idea 1)
  misalign_lang_preserve_md   target=LANG, preserve=MD       (mirror)
  misalign_all                target=LANG+MD, no preserve    (anchor replicate)
  preserve_control            same heads, fMRI rows PERMUTED (anchor "Brain Preserving")
  brain_tuned_md              target head without GRL (push MD alignment UP)
  lm_only                     only L_lm (stimulus-tuned control)

Options that address "shared features will drag the preserved network down":
  --orthogonalize   : the target head is trained on the MD signal *residualised
                      on LANG BOLD* (computed in 03) instead of raw MD.
  --grad_surgery    : PCGrad-style -- project the GRL branch gradient onto the
                      orthogonal complement of the preserve-branch gradient
                      (per LoRA parameter) when they conflict.

Usage
-----
  python 04_train_region_misalign.py --prepared data/prepared --model gpt2 \
      --condition misalign_md_preserve_lang --heldout_story 0 --out runs/misMD_s0
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoModelForMaskedLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import PreparedDataset, get_grl, neg_pearson_loss, pearson_cols, zscore  # noqa: E402

grad_reverse = get_grl()


# --------------------------------------------------------------------------- #
class TRDataset(Dataset):
    def __init__(self, texts, Y, idx):
        self.texts = [texts[i] for i in idx]
        self.Y = torch.tensor(Y[idx], dtype=torch.float32)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, i):
        return self.texts[i], self.Y[i]


def collate(batch, tok, max_len, is_mlm, mlm_prob=0.15):
    texts, Y = zip(*batch)
    enc = tok(list(texts), return_tensors="pt", padding=True, truncation=True, max_length=max_len)
    labels = enc["input_ids"].clone()
    if is_mlm:
        # standard BERT masking
        prob = torch.full(labels.shape, mlm_prob)
        special = torch.tensor([[t in tok.all_special_ids for t in row] for row in labels.tolist()])
        prob.masked_fill_(special, 0.0)
        masked = torch.bernoulli(prob).bool()
        labels[~masked] = -100
        enc["input_ids"][masked] = tok.mask_token_id
    else:
        labels[enc["attention_mask"] == 0] = -100
    enc["labels"] = labels
    return enc, torch.stack(Y)


class BrainHeads(nn.Module):
    def __init__(self, d, n_target, n_preserve):
        super().__init__()
        self.target = nn.Linear(d, n_target) if n_target > 0 else None
        self.preserve = nn.Linear(d, n_preserve) if n_preserve > 0 else None


def pooled_hidden(out, attention_mask):
    h = out.hidden_states[-1]
    m = attention_mask.unsqueeze(-1).float()
    return (h * m).sum(1) / m.sum(1).clamp(min=1)


def pcgrad_project(params, loss_target, loss_preserve):
    """Two backward passes; project target-grad away from preserve-grad when
    they conflict (dot < 0). Returns combined grads set on params."""
    g_t = torch.autograd.grad(loss_target, params, retain_graph=True, allow_unused=True)
    g_p = torch.autograd.grad(loss_preserve, params, retain_graph=True, allow_unused=True)
    for p, gt, gp in zip(params, g_t, g_p):
        if gt is None and gp is None:
            continue
        gt = torch.zeros_like(p) if gt is None else gt
        gp = torch.zeros_like(p) if gp is None else gp
        dot = (gt * gp).sum()
        if dot < 0:
            gt = gt - dot / (gp.norm() ** 2 + 1e-12) * gp
        p.grad = (gt + gp) if p.grad is None else p.grad + gt + gp


# --------------------------------------------------------------------------- #
def main(a):
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ds = PreparedDataset.load(a.prepared)
    Y = zscore(ds.Y)
    iL, iM = ds.idx("LANG"), ds.idx("MD")

    # ---- choose target / preserve channels per condition -------------------
    cond = a.condition
    grl_on, permute = True, False
    if cond == "misalign_md_preserve_lang":
        tgt, prs = iM, iL
    elif cond == "misalign_lang_preserve_md":
        tgt, prs = iL, iM
    elif cond == "misalign_all":
        tgt, prs = np.concatenate([iL, iM]), np.array([], int)
    elif cond == "preserve_control":
        tgt, prs = iM, iL
        permute = True
    elif cond == "brain_tuned_md":
        tgt, prs = iM, iL
        grl_on = False
    elif cond == "lm_only":
        tgt, prs = np.array([], int), np.array([], int)
    else:
        raise SystemExit(f"unknown condition {cond}")

    Yt = Y[:, tgt] if len(tgt) else np.zeros((Y.shape[0], 0))
    Yp = Y[:, prs] if len(prs) else np.zeros((Y.shape[0], 0))
    if a.orthogonalize and len(tgt) and len(prs):
        # residualise target on preserve BOLD (story-wise OLS, closed form)
        from common import story_cv_predict
        Yt = Yt - story_cv_predict(Yp, Yt, ds.story_ids, alphas=(0.1, 1, 10))
        Yt = zscore(Yt)
    if permute:
        rng = np.random.default_rng(a.seed)
        for s in np.unique(ds.story_ids):
            ix = np.where(ds.story_ids == s)[0]
            Yt[ix] = Yt[rng.permutation(ix)]
            if len(prs):
                Yp[ix] = Yp[rng.permutation(ix)]
    Yall = np.hstack([Yt, Yp])
    n_t, n_p = Yt.shape[1], Yp.shape[1]

    train_idx = np.where(ds.story_ids != a.heldout_story)[0]
    test_idx = np.where(ds.story_ids == a.heldout_story)[0]

    # ---- model -------------------------------------------------------------
    tok = AutoTokenizer.from_pretrained(a.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token or tok.unk_token
    is_mlm = "bert" in a.model.lower()
    base = (AutoModelForMaskedLM if is_mlm else AutoModelForCausalLM).from_pretrained(
        a.model, torch_dtype=torch.float32)
    target_modules = a.lora_targets.split(",") if a.lora_targets else None
    lcfg = LoraConfig(r=a.lora_r, lora_alpha=2 * a.lora_r, lora_dropout=0.05,
                      target_modules=target_modules, task_type=None)
    model = get_peft_model(base, lcfg).to(device)
    d = model.config.hidden_size
    heads = BrainHeads(d, n_t, n_p).to(device)
    lora_params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(list(lora_params) + list(heads.parameters()), lr=a.lr, weight_decay=0.01)

    dl = DataLoader(TRDataset(ds.text_windows, Yall, train_idx), batch_size=a.batch_size, shuffle=True,
                    collate_fn=lambda b: collate(b, tok, a.max_len, is_mlm))
    steps = a.epochs * len(dl)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 1 - s / max(1, steps))
    log = []
    print(f"condition={cond} target={n_t}ch preserve={n_p}ch grl={grl_on} permute={permute} "
          f"orth={a.orthogonalize} surgery={a.grad_surgery} train={len(train_idx)} test={len(test_idx)}")

    model.train()
    t0 = time.time()
    for ep in range(a.epochs):
        for step, (enc, yb) in enumerate(dl):
            enc = {k: v.to(device) for k, v in enc.items()}
            yb = yb.to(device)
            out = model(**enc, output_hidden_states=True)
            l_lm = out.loss
            h = pooled_hidden(out, enc["attention_mask"])
            l_t = torch.tensor(0.0, device=device)
            l_p = torch.tensor(0.0, device=device)
            if n_t:
                ht = grad_reverse(h, a.grl_lambda) if grl_on else h
                l_t = neg_pearson_loss(heads.target(ht), yb[:, :n_t], squared=grl_on)
            if n_p:
                l_p = neg_pearson_loss(heads.preserve(h), yb[:, n_t:], squared=False)
            opt.zero_grad()
            if a.grad_surgery and n_t and n_p:
                (a.w_lm * l_lm).backward(retain_graph=True)
                pcgrad_project(lora_params, a.w_target * l_t, a.w_preserve * l_p)
                # heads get plain grads
                (a.w_target * l_t + a.w_preserve * l_p).backward(inputs=list(heads.parameters()))
            else:
                loss = a.w_lm * l_lm + a.w_target * l_t + a.w_preserve * l_p
                loss.backward()
            torch.nn.utils.clip_grad_norm_(list(lora_params) + list(heads.parameters()), 1.0)
            opt.step()
            sched.step()
            if step % 20 == 0:
                rec = {"epoch": ep, "step": step, "l_lm": float(l_lm), "l_target": float(l_t),
                       "l_preserve": float(l_p), "t": time.time() - t0}
                log.append(rec)
                print(rec)

    # ---- quick held-out check with the trained heads -----------------------
    model.eval()
    with torch.no_grad():
        dlt = DataLoader(TRDataset(ds.text_windows, Yall, test_idx), batch_size=a.batch_size,
                         collate_fn=lambda b: collate(b, tok, a.max_len, is_mlm))
        preds, ys, lm_losses = [], [], []
        for enc, yb in dlt:
            enc = {k: v.to(device) for k, v in enc.items()}
            out = model(**enc, output_hidden_states=True)
            lm_losses.append(float(out.loss))
            h = pooled_hidden(out, enc["attention_mask"])
            p = []
            if n_t:
                p.append(heads.target(h))
            if n_p:
                p.append(heads.preserve(h))
            if p:
                preds.append(torch.cat(p, 1).cpu().numpy())
            ys.append(yb.numpy())
        summary = {"heldout_story": a.heldout_story, "heldout_lm_loss": float(np.mean(lm_losses)),
                   "heldout_ppl": float(math.exp(np.mean(lm_losses)))}
        if preds:
            P, Yh = np.vstack(preds), np.vstack(ys)
            r = pearson_cols(Yh, P)
            summary["head_r_target_mean"] = float(r[:n_t].mean()) if n_t else None
            summary["head_r_preserve_mean"] = float(r[n_t:].mean()) if n_p else None
    print("HELD-OUT:", summary)

    os.makedirs(a.out, exist_ok=True)
    model.save_pretrained(os.path.join(a.out, "adapter"))
    torch.save(heads.state_dict(), os.path.join(a.out, "heads.pt"))
    with open(os.path.join(a.out, "config.json"), "w") as f:
        json.dump({**vars(a), "n_target": n_t, "n_preserve": n_p,
                   "target_frois": [ds.froi_names[i] for i in tgt],
                   "preserve_frois": [ds.froi_names[i] for i in prs]}, f, indent=2)
    with open(os.path.join(a.out, "train_log.json"), "w") as f:
        json.dump({"log": log, "heldout": summary}, f, indent=2)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepared", required=True)
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--condition", required=True,
                    choices=["misalign_md_preserve_lang", "misalign_lang_preserve_md", "misalign_all",
                             "preserve_control", "brain_tuned_md", "lm_only"])
    ap.add_argument("--heldout_story", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=5)          # anchor: 5 epochs
    ap.add_argument("--batch_size", type=int, default=16)     # anchor: 16
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--max_len", type=int, default=128)
    ap.add_argument("--w_lm", type=float, default=0.1)        # anchor: 0.1
    ap.add_argument("--w_target", type=float, default=10.0)   # anchor: 10
    ap.add_argument("--w_preserve", type=float, default=10.0)
    ap.add_argument("--grl_lambda", type=float, default=1.0)
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_targets", default=None,
                    help="comma list, e.g. 'c_attn,c_proj' (gpt2) or 'q_proj,k_proj,v_proj,o_proj' (llama)")
    ap.add_argument("--orthogonalize", action="store_true")
    ap.add_argument("--grad_surgery", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    main(ap.parse_args())
