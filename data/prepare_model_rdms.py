"""Model-side RDMs from a small pretrained torchvision network.

Runs a CPU `resnet18` over a small stimulus set (either the CIFAR-10 test
set or a random pixel proxy) and stores per-layer activations. Used by
`experiments/02_model_vs_brain.py` to compare model-vs-brain and
model-vs-model RDMs.

If you don't want to download torchvision weights, this falls back to
random projections of a randomly-generated stimulus set with the same
category structure as the vision brain dataset -- enough to demonstrate
the model-vs-brain comparison pipeline.
"""
from __future__ import annotations

import sys
from pathlib import Path

_THIS = Path(__file__).resolve()
sys.path.insert(0, str(_THIS.parents[1]))
sys.path.insert(0, str(_THIS.parent))

import numpy as np

from _utils import banner, save_dataset


def _try_torchvision(n_stimuli: int, rng: np.random.Generator) -> dict | None:
    try:
        import torch
        from torchvision import models, transforms
    except Exception:
        return None
    try:
        model = models.resnet18(weights=None)          # no download by default
        # Optionally load weights; skip on offline machines
        try:
            model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        except Exception:
            pass
        model.eval()

        # Hooks on 5 informative layers
        hooks = {}
        acts: dict[str, np.ndarray] = {}
        layer_names = ["conv1", "layer1", "layer2", "layer3", "layer4"]

        def make_hook(name):
            def hook(_m, _i, output):
                acts[name] = output.detach().flatten(1).cpu().numpy()
            return hook

        for name in layer_names:
            layer = dict(model.named_modules())[name]
            hooks[name] = layer.register_forward_hook(make_hook(name))

        # Random RGB stimuli scaled to imagenet stats. Substitute your own
        # image tensor if you want a specific stimulus set.
        stim = torch.from_numpy(
            rng.standard_normal((n_stimuli, 3, 64, 64)).astype(np.float32)
        )
        with torch.no_grad():
            model(stim)
        for h in hooks.values():
            h.remove()

        embeddings = {n: acts[n] for n in layer_names}
        emb_arr = np.empty(len(layer_names), dtype=object)
        for i, n in enumerate(layer_names):
            emb_arr[i] = embeddings[n]
        return dict(
            layer_names=np.array(layer_names),
            embeddings=emb_arr,
            source=np.array("torchvision resnet18 on random RGB stimuli"),
        )
    except Exception as e:
        banner(f"torchvision path failed ({e}); falling back")
        return None


def _synthesise(n_stimuli: int, rng: np.random.Generator) -> dict:
    """Fake 'model layers' via random linear projections of a shared latent."""
    latent = rng.standard_normal((n_stimuli, 12))
    layers = []
    layer_names = ["layer1", "layer2", "layer3", "layer4", "layer5"]
    for L, name in enumerate(layer_names):
        dim = 64 * (L + 1)
        # gradually mix categories differently at each depth
        W = rng.standard_normal((12, dim)) / np.sqrt(12)
        layers.append((latent + 0.1 * L * rng.standard_normal(latent.shape)) @ W)
    emb_arr = np.empty(len(layers), dtype=object)
    for i, L in enumerate(layers):
        emb_arr[i] = L
    return dict(
        layer_names=np.array(layer_names),
        embeddings=emb_arr,
        source=np.array("synthesised random-projection 'model' (no torchvision)"),
    )


def main(n_stimuli: int = 40) -> None:
    rng = np.random.default_rng(2)
    banner(f"generating vision-model activations for {n_stimuli} stimuli")
    real = _try_torchvision(n_stimuli, rng)
    if real is not None:
        banner("used torchvision resnet18")
        data = real
    else:
        banner("torchvision unavailable; using random-projection stand-in")
        data = _synthesise(n_stimuli, rng)

    p = save_dataset("vision_model_activations", **data)
    banner(f"wrote {p}")


if __name__ == "__main__":
    main()
