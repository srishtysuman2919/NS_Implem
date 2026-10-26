#!/usr/bin/env python
"""
Step 6 -- Linguistic-competence evaluation of the (mis)aligned models.

Fast, zero-shot, no-training evaluations suitable for a same-day POC:
  * BLiMP (Warstadt et al. 2020): 67 minimal-pair paradigms grouped into 12
    linguistic phenomena. Score = fraction of pairs where the grammatical
    sentence gets higher total log-probability.  Causal LMs only.
  * Held-out story perplexity (language modelling sanity check).

For the full 200+ probing-dataset Holmes benchmark used by the anchor paper,
run the official FlashHolmes suite on each saved adapter afterwards:
    https://github.com/UKPLab/holmes   (Waldis et al., TACL 2024)
Point its model loader at the merged checkpoint written by --save_merged.

Usage
-----
  python 06_eval_linguistic.py --model gpt2 --adapter runs/misMD_s0/adapter \
      --prepared data/prepared --heldout_story 0 --out results/ling_misMD_s0.json
  python 06_eval_linguistic.py --model gpt2 --out results/ling_base.json  # base model
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np
import torch
from datasets import get_dataset_config_names, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import PreparedDataset  # noqa: E402

# BLiMP paradigm -> phenomenon (12 groups), mirrors the official grouping
PHENOMENA = {
    "anaphor_agreement": ["anaphor_gender_agreement", "anaphor_number_agreement"],
    "argument_structure": ["animate_subject_passive", "animate_subject_trans", "causative", "drop_argument",
                           "inchoative", "intransitive", "passive_1", "passive_2", "transitive"],
    "binding": ["principle_A_c_command", "principle_A_case_1", "principle_A_case_2", "principle_A_domain_1",
                "principle_A_domain_2", "principle_A_domain_3", "principle_A_reconstruction"],
    "control_raising": ["existential_there_object_raising", "existential_there_subject_raising",
                        "expletive_it_object_raising", "tough_vs_raising_1", "tough_vs_raising_2"],
    "determiner_noun_agreement": ["determiner_noun_agreement_1", "determiner_noun_agreement_2",
                                  "determiner_noun_agreement_irregular_1", "determiner_noun_agreement_irregular_2",
                                  "determiner_noun_agreement_with_adj_2", "determiner_noun_agreement_with_adj_irregular_1",
                                  "determiner_noun_agreement_with_adj_irregular_2", "determiner_noun_agreement_with_adjective_1"],
    "ellipsis": ["ellipsis_n_bar_1", "ellipsis_n_bar_2"],
    "filler_gap": ["wh_questions_object_gap", "wh_questions_subject_gap", "wh_questions_subject_gap_long_distance",
                   "wh_vs_that_no_gap", "wh_vs_that_no_gap_long_distance", "wh_vs_that_with_gap",
                   "wh_vs_that_with_gap_long_distance"],
    "irregular_forms": ["irregular_past_participle_adjectives", "irregular_past_participle_verbs"],
    "island_effects": ["adjunct_island", "complex_NP_island", "coordinate_structure_constraint_complex_left_branch",
                       "coordinate_structure_constraint_object_extraction", "left_branch_island_echo_question",
                       "left_branch_island_simple_question", "sentential_subject_island", "wh_island"],
    "npi_licensing": ["matrix_question_npi_licensor_present", "npi_present_1", "npi_present_2",
                      "only_npi_licensor_present", "only_npi_scope", "sentential_negation_npi_licensor_present",
                      "sentential_negation_npi_scope"],
    "quantifiers": ["existential_there_quantifiers_1", "existential_there_quantifiers_2",
                    "superlative_quantifiers_1", "superlative_quantifiers_2"],
    "subject_verb_agreement": ["distractor_agreement_relational_noun", "distractor_agreement_relative_clause",
                               "irregular_plural_subject_verb_agreement_1", "irregular_plural_subject_verb_agreement_2",
                               "regular_plural_subject_verb_agreement_1", "regular_plural_subject_verb_agreement_2"],
}


@torch.no_grad()
def seq_logprob(model, tok, sents, device, bs=32):
    out = []
    for i in range(0, len(sents), bs):
        enc = tok(sents[i:i + bs], return_tensors="pt", padding=True).to(device)
        logits = model(**enc).logits[:, :-1]
        tgt = enc["input_ids"][:, 1:]
        mask = enc["attention_mask"][:, 1:].float()
        lp = torch.log_softmax(logits.float(), -1).gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
        out.append((lp * mask).sum(1).cpu().numpy())
    return np.concatenate(out)


def blimp(model, tok, device, max_per_task=None):
    res = {}
    for cfg in get_dataset_config_names("blimp"):
        d = load_dataset("blimp", cfg, split="train")
        if max_per_task:
            d = d.select(range(min(max_per_task, len(d))))
        g = seq_logprob(model, tok, d["sentence_good"], device)
        b = seq_logprob(model, tok, d["sentence_bad"], device)
        res[cfg] = float((g > b).mean())
        print(f"  {cfg:60s} {res[cfg]:.3f}")
    by_phen = {p: float(np.mean([res[t] for t in ts if t in res])) for p, ts in PHENOMENA.items()}
    return {"per_paradigm": res, "per_phenomenon": by_phen, "overall": float(np.mean(list(res.values())))}


@torch.no_grad()
def story_ppl(model, tok, texts, device, bs=16):
    nll, n = 0.0, 0
    for i in range(0, len(texts), bs):
        enc = tok(texts[i:i + bs], return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
        labels = enc["input_ids"].clone()
        labels[enc["attention_mask"] == 0] = -100
        out = model(**enc, labels=labels)
        k = int((labels != -100).sum())
        nll += float(out.loss) * k
        n += k
    return math.exp(nll / max(n, 1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--prepared", default=None)
    ap.add_argument("--heldout_story", type=int, default=None)
    ap.add_argument("--max_per_task", type=int, default=None, help="subsample BLiMP for speed (e.g. 300)")
    ap.add_argument("--save_merged", default=None, help="dir to save merged model for Holmes")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(a.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.float32)
    if a.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, a.adapter).merge_and_unload()
    model.eval().to(device)
    if a.save_merged:
        model.save_pretrained(a.save_merged)
        tok.save_pretrained(a.save_merged)

    result = {"model": a.model, "adapter": a.adapter}
    if a.prepared is not None and a.heldout_story is not None:
        ds = PreparedDataset.load(a.prepared)
        texts = [t for t, s in zip(ds.text_windows, ds.story_ids) if s == a.heldout_story]
        result["heldout_story_ppl"] = story_ppl(model, tok, texts, device)
        print("held-out story ppl:", result["heldout_story_ppl"])
    print("BLiMP:")
    result["blimp"] = blimp(model, tok, device, a.max_per_task)
    print("BLiMP overall:", result["blimp"]["overall"])
    print(json.dumps(result["blimp"]["per_phenomenon"], indent=2))
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(result, open(a.out, "w"), indent=2)
