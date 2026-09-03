"""LLM ↔ brain representational alignment
-- in the spirit of Jozwik et al. (2025), Schrimpf et al. (2021),
Caucheteux & King (2022), Toneva & Wehbe (2019), and related work.

Paper goal
----------
Do the internal representations of pretrained language models look like
the language cortex's representations of the same stimuli? The
community-wide finding is that **middle transformer layers align best**
with high-level language cortex, both under RSA and under ridge-encoding
metrics; earlier layers look like phonology / syntax, later layers look
like task-specific readouts. Jozwik et al. 2025 sharpen this by comparing
alignment across a broad panel of LMs and brain datasets.

Contents of this module
-----------------------
* :func:`get_layerwise_embeddings` -- for a list of sentences, extract
  per-layer hidden-state embeddings from a HuggingFace causal LM. Pools
  over tokens (mean) so each sentence becomes one vector per layer.
* :func:`layerwise_rsa`            -- for each layer, build the model RDM
  and Spearman-compare to a given brain RDM.
* :func:`layerwise_cka`            -- linear CKA per layer against brain
  features.
* :func:`layerwise_ridge`          -- k-fold ridge encoding score per layer.

Everything runs on CPU with a small model (`gpt2` by default) so it fits
in a study repo.

What shortcoming of the previous method does this address?
---------------------------------------------------------
Everything above was demonstrated on vision data. This module shows that
the **same toolkit is directly applicable to language-NeuroAI**: swap the
stimulus set (images → sentences) and the model (CNN → transformer) and
all the RSA/CKA/ridge routines apply unchanged. The layer-wise curve is
what makes this useful for actually *interpreting* language models
against the brain, rather than reporting a single similarity number.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np

from .classic_rsa import rdm, rdm_similarity
from .cka import linear_cka
from .ridge_encoding import ridge_cv_score


# ---------------------------------------------------------------------------
# Embedding extraction
# ---------------------------------------------------------------------------
def get_layerwise_embeddings(
    sentences: list[str],
    model_name: str = "gpt2",
    pool: str = "mean",
    device: str = "cpu",
    max_length: int = 64,
) -> tuple[np.ndarray, list[str]]:
    """Extract per-layer sentence embeddings from a HuggingFace causal LM.

    Parameters
    ----------
    sentences : list[str]
    model_name : str
        Any HuggingFace causal LM small enough for CPU: ``gpt2`` (124M) is
        the default. ``distilgpt2`` is even smaller if needed.
    pool : {'mean', 'last'}
        Token-pooling strategy. ``mean`` is more RSA-friendly; ``last`` is
        common in causal-LM encoding-model papers because the last token
        has attended to the whole sentence.
    device : str
    max_length : int
        Truncate sentences to this many tokens.

    Returns
    -------
    embeddings : (n_layers, n_sentences, hidden_dim) ndarray
        Includes layer 0 (the embedding layer) plus each transformer block
        output.
    layer_names : list[str]
    """
    try:
        import torch
        from transformers import AutoModel, AutoTokenizer
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "transformers + torch are required for LLM alignment. "
            "pip install torch transformers"
        ) from e

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModel.from_pretrained(model_name, output_hidden_states=True)
    model.eval()
    model.to(device)

    all_layers = None
    with torch.no_grad():
        for i, sent in enumerate(sentences):
            enc = tokenizer(
                sent,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
                padding=False,
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            out = model(**enc)
            hs = out.hidden_states                    # tuple of (1, T, H)
            if all_layers is None:
                n_layers = len(hs)
                hidden_dim = hs[0].shape[-1]
                all_layers = np.zeros(
                    (n_layers, len(sentences), hidden_dim), dtype=np.float32
                )
            for L, h in enumerate(hs):
                v = h[0]                              # (T, H)
                if pool == "mean":
                    vec = v.mean(dim=0).cpu().numpy()
                elif pool == "last":
                    vec = v[-1].cpu().numpy()
                else:
                    raise ValueError(f"unknown pool {pool!r}")
                all_layers[L, i] = vec

    layer_names = [f"embed"] + [f"h{L}" for L in range(1, all_layers.shape[0])]
    return all_layers, layer_names


# ---------------------------------------------------------------------------
# Layer-wise comparisons
# ---------------------------------------------------------------------------
def layerwise_rsa(
    embeddings: np.ndarray,
    brain_rdm: np.ndarray,
    metric: str = "correlation",
    compare: str = "spearman",
) -> np.ndarray:
    """Spearman ρ per layer between model RDM and ``brain_rdm``."""
    n_layers = embeddings.shape[0]
    out = np.zeros(n_layers)
    for L in range(n_layers):
        model_rdm = rdm(embeddings[L], metric=metric)
        out[L] = rdm_similarity(model_rdm, brain_rdm, method=compare)
    return out


def layerwise_cka(
    embeddings: np.ndarray,
    brain_features: np.ndarray,
) -> np.ndarray:
    """Linear CKA per layer against ``brain_features``.

    ``brain_features`` is a ``(n_stimuli, n_voxels)`` matrix -- typically the
    ROI-restricted response matrix.
    """
    n_layers = embeddings.shape[0]
    out = np.zeros(n_layers)
    for L in range(n_layers):
        out[L] = linear_cka(embeddings[L], brain_features)
    return out


def layerwise_ridge(
    embeddings: np.ndarray,
    brain_features: np.ndarray,
    alphas: Iterable[float] = (1.0, 10.0, 100.0, 1000.0),
    k: int = 5,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Cross-validated ridge encoding score, per layer, averaged over voxels."""
    rng = rng or np.random.default_rng(0)
    n_layers = embeddings.shape[0]
    out = np.zeros(n_layers)
    for L in range(n_layers):
        res = ridge_cv_score(
            embeddings[L],
            brain_features,
            alphas=list(alphas),
            k=k,
            rng=rng,
            select_alpha_per_output=False,
        )
        out[L] = np.nanmean(res["scores"])
    return out


# ---------------------------------------------------------------------------
# self-test (offline; skipped if transformers is not installed)
# ---------------------------------------------------------------------------
def _selftest() -> None:
    try:
        from transformers import AutoTokenizer  # noqa: F401
    except ImportError:
        print("llm_alignment self-test skipped (transformers not installed)")
        return
    sentences = [
        "The cat sat on the mat.",
        "A dog barked loudly.",
        "Language models predict the next token.",
        "Blue whales are the largest animals.",
        "Music can influence emotions strongly.",
        "Rain fell throughout the night.",
    ]
    emb, names = get_layerwise_embeddings(sentences, model_name="distilgpt2")
    print(f"llm_alignment self-test OK")
    print(f"  n_layers          = {emb.shape[0]}   ({names[0]}..{names[-1]})")
    print(f"  hidden dim        = {emb.shape[2]}")
    # fake a brain RDM by shuffling the middle-layer RDM slightly
    mid = emb.shape[0] // 2
    fake_brain = rdm(emb[mid] + 0.1 * np.random.default_rng(0).standard_normal(
        emb[mid].shape).astype(np.float32))
    curve = layerwise_rsa(emb, fake_brain)
    print(f"  layer-wise RSA curve (peaks near layer {curve.argmax()}):")
    for L, r in enumerate(curve):
        print(f"    {names[L]:>5s}  ρ = {r:+.3f}")


if __name__ == "__main__":
    _selftest()
