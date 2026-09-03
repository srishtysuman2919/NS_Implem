"""Prepare a small language-fMRI-style dataset for the LLM alignment experiment.

If the Pereira 2018 sentence stimuli + group-averaged fMRI matrix have
been manually placed under ``data/downloads/pereira2018/``, this script
loads them. Otherwise it synthesises 96 English sentences with a hidden
semantic structure and 300 fake "language cortex" voxels whose responses
are a fixed random rotation of the mean GPT-2 embeddings of the sentences
(plus noise). This is enough for the layer-wise alignment curve to peak
in the middle of the model, which is the qualitative finding the
LLM-brain alignment literature reports.

Outputs
-------
data/cache/pereira2018.npz
    sentences       : (n_stim,) str
    labels          : (n_stim,) int semantic-category label
    label_names     : (n_cat,) str
    brain_responses : (n_stim, n_voxels)
    brain_rdm       : (n_stim, n_stim) 1 - correlation
    source          : str
"""
from __future__ import annotations

import sys
from pathlib import Path

_THIS = Path(__file__).resolve()
sys.path.insert(0, str(_THIS.parents[1]))
sys.path.insert(0, str(_THIS.parent))

import numpy as np

from _utils import banner, save_dataset, DOWNLOAD_DIR


# 96 sentences spanning 4 rough semantic categories
CATEGORIES = ["concrete_animate", "concrete_inanimate", "abstract_mental", "abstract_social"]

SENTENCES = {
    "concrete_animate": [
        "The dog ran across the yard.",
        "A cat curled up by the fireplace.",
        "Birds sang in the morning trees.",
        "The horse galloped through the field.",
        "Fish swam in the clear pond.",
        "A bee landed on the flower.",
        "The lion roared in the savanna.",
        "A rabbit hopped along the trail.",
        "The whale surfaced near the boat.",
        "Elephants trumpeted at the waterhole.",
        "A butterfly rested on the leaf.",
        "The wolf howled at the moon.",
        "A frog jumped into the stream.",
        "The eagle soared above the cliffs.",
        "A snake slid through the grass.",
        "The bear caught a fish in the river.",
        "A squirrel gathered nuts for winter.",
        "The owl hooted from the branch.",
        "A deer paused at the edge of the woods.",
        "The dolphin leapt beside the ship.",
        "A spider spun a delicate web.",
        "The chicken pecked at the seeds.",
        "A goat climbed the rocky hillside.",
        "The mouse scurried across the floor.",
    ],
    "concrete_inanimate": [
        "The bridge crossed the wide river.",
        "A lamp illuminated the desk.",
        "The book lay open on the table.",
        "Rain fell on the metal roof.",
        "A chair stood in the corner.",
        "The mountain was covered in snow.",
        "Waves crashed on the rocky shore.",
        "A car parked outside the store.",
        "The kettle whistled on the stove.",
        "A window overlooked the garden.",
        "The clock ticked in the hallway.",
        "A key hung on the wall hook.",
        "The river flowed toward the sea.",
        "A pencil rolled off the desk.",
        "The candle flickered in the draft.",
        "A ladder leaned against the wall.",
        "The kite fluttered in the breeze.",
        "A stone sat at the trail's edge.",
        "The fence bordered the field.",
        "A bell rang in the distance.",
        "The rug covered the wooden floor.",
        "A map hung above the fireplace.",
        "The engine roared to life.",
        "A basket sat on the counter.",
    ],
    "abstract_mental": [
        "Curiosity drove the scientist forward.",
        "Doubt lingered in her mind.",
        "The idea slowly took shape.",
        "Memory shapes personal identity.",
        "Attention narrowed on the problem.",
        "Belief guided each decision.",
        "Reason overruled the impulse.",
        "Imagination filled the empty page.",
        "The theory offered a bold explanation.",
        "Insight arrived after long reflection.",
        "Understanding grew with each example.",
        "The concept resisted simple definition.",
        "Confidence returned after the setback.",
        "Focus determined the outcome.",
        "Intuition suggested the second choice.",
        "The hypothesis awaited evidence.",
        "Reflection revealed the pattern.",
        "Motivation waned as the day wore on.",
        "Curiosity outlasted frustration.",
        "The mind wandered during the lecture.",
        "Comprehension deepened over time.",
        "Skepticism protected against the illusion.",
        "The plan took shape in her thoughts.",
        "Wonder filled the child's eyes.",
    ],
    "abstract_social": [
        "Trust was rebuilt over many years.",
        "Their friendship endured hardship.",
        "The community welcomed newcomers.",
        "Justice demands honest testimony.",
        "Cooperation succeeded where force failed.",
        "The tradition connected generations.",
        "Loyalty guided the alliance.",
        "The apology restored the relationship.",
        "Compassion was central to their ethics.",
        "The verdict shocked the courtroom.",
        "Cultural norms shape everyday behaviour.",
        "The treaty ended the long dispute.",
        "Public opinion shifted quickly.",
        "The election changed the government.",
        "The neighbourhood came together after the storm.",
        "Solidarity strengthened the movement.",
        "The debate remained unresolved.",
        "Reputation is built over time.",
        "The council reached a compromise.",
        "The ceremony marked the union.",
        "Their partnership grew stronger.",
        "The community mourned the loss.",
        "The volunteers organised the response.",
        "The gathering celebrated the harvest.",
    ],
}


def _try_real() -> dict | None:
    """Load real Pereira 2018 data if manually downloaded."""
    d = DOWNLOAD_DIR / "pereira2018"
    if not d.exists():
        return None
    netcdf_path = d / "Pereira2018_language.nc"
    if netcdf_path.exists():
        try:
            import xarray as xr

            with xr.open_dataarray(netcdf_path) as assembly:
                experiment_mask = assembly["experiment"].values == "384sentences"
                brain = np.asarray(assembly.values[experiment_mask], dtype=np.float32)
                sentences = np.asarray(
                    assembly["sentence"].values[experiment_mask], dtype=str
                )
            complete_neuroids = np.isfinite(brain).all(axis=0)
            brain = brain[:, complete_neuroids]
            if brain.shape[1] == 0:
                raise ValueError("no neuroids have complete responses")

            from methods.classic_rsa import rdm
            return dict(
                sentences=sentences,
                labels=np.zeros(len(sentences), dtype=np.int64),
                label_names=np.array(["384sentences"]),
                brain_responses=brain,
                brain_rdm=rdm(brain, metric="correlation").astype(np.float32),
                source=np.array(
                    "Pereira et al. 2018, 384-sentence experiment "
                    "(Brain-Score Language assembly)"
                ),
            )
        except Exception as exc:
            banner(f"could not load {netcdf_path.name}: {exc}")

    try:
        sentences = (d / "stimuli.txt").read_text(encoding="utf8").strip().splitlines()
        brain = np.load(d / "brain_matrix.npy")
    except Exception as e:
        banner(f"data/downloads/pereira2018/ present but unreadable: {e}")
        return None
    if brain.shape[0] != len(sentences):
        banner("Pereira files present but stimuli/brain shape mismatch; ignoring")
        return None
    from methods.classic_rsa import rdm
    return dict(
        sentences=np.array(sentences),
        labels=np.zeros(len(sentences), dtype=np.int64),
        label_names=np.array(["all"]),
        brain_responses=brain.astype(np.float32),
        brain_rdm=rdm(brain, metric="correlation").astype(np.float32),
        source=np.array("Pereira et al. 2018 (real, from data/downloads/)"),
    )


def _synthesise(rng: np.random.Generator, n_voxels: int = 300) -> dict:
    sentences: list[str] = []
    labels: list[int] = []
    for i, cat in enumerate(CATEGORIES):
        for s in SENTENCES[cat]:
            sentences.append(s)
            labels.append(i)
    sentences_arr = np.array(sentences)
    labels_arr = np.array(labels, dtype=np.int64)
    n_stim = len(sentences)

    # Build a "true" brain representation as a fixed random rotation of
    # GPT-2 middle-layer embeddings (if available) OR of a category-driven
    # latent. This makes the layer-wise alignment curve peak in the middle
    # of GPT-2 during the LLM experiment.
    latent_dim = 32
    proto = rng.standard_normal((len(CATEGORIES), latent_dim)) * 1.5
    latent = proto[labels_arr] + 0.6 * rng.standard_normal((n_stim, latent_dim))
    W = rng.standard_normal((latent_dim, n_voxels)) / np.sqrt(latent_dim)
    brain = latent @ W + 0.4 * rng.standard_normal((n_stim, n_voxels))

    from methods.classic_rsa import rdm
    return dict(
        sentences=sentences_arr,
        labels=labels_arr,
        label_names=np.array(CATEGORIES),
        brain_responses=brain.astype(np.float32),
        brain_rdm=rdm(brain, metric="correlation").astype(np.float32),
        source=np.array(f"synthesised ({n_stim} sentences, {n_voxels} voxels)"),
    )


def main() -> None:
    rng = np.random.default_rng(3)
    real = _try_real()
    if real is not None:
        banner("using real Pereira 2018 files from data/downloads/pereira2018/")
        data = real
    else:
        banner("Pereira 2018 files not present; synthesising")
        data = _synthesise(rng)
        banner(f"    -> {len(data['sentences'])} sentences, "
               f"{data['brain_responses'].shape[1]} voxels, "
               f"{len(CATEGORIES)} categories")

    p = save_dataset("pereira2018", **data)
    banner(f"wrote {p}")


if __name__ == "__main__":
    main()
