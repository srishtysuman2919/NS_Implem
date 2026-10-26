#!/usr/bin/env bash
# End-to-end POC for one base model and one held-out story.
#   bash run_all.sh gpt2 12 0          # model, feature layer (hidden_states idx), held-out story
# Stage A (gate) is CPU-friendly; Stage B needs a GPU (gpt2: ~10-20 min/condition on one GPU).
set -euo pipefail
MODEL=${1:-gpt2}
LAYER=${2:-12}
STORY=${3:-0}
TAG=$(echo "$MODEL" | tr '/' '_')
PREP=data/prepared

# ---------------- Stage A: GO / NO-GO gate (no training) --------------------
python 02_extract_features.py --prepared $PREP --model $MODEL --layers $LAYER --out features/${TAG}_base
python 03_gate_analysis.py --prepared $PREP --features features/${TAG}_base/layer${LAYER}.npy \
       --out results/gate_${TAG}_L${LAYER}.json
echo ">>> Read results/gate_${TAG}_L${LAYER}.json  -> 'verdict' before continuing."
read -p "Continue to Stage B (training)? [y/N] " yn
[[ "$yn" == "y" ]] || exit 0

# ---------------- Stage B: interventions ------------------------------------
LORA_T=""; [[ "$MODEL" == gpt2* ]] && LORA_T="--lora_targets c_attn,c_proj"
[[ "$MODEL" == *lama* ]] && LORA_T="--lora_targets q_proj,k_proj,v_proj,o_proj"

for COND in preserve_control misalign_md_preserve_lang misalign_lang_preserve_md misalign_all lm_only; do
  python 04_train_region_misalign.py --prepared $PREP --model $MODEL --condition $COND \
         --heldout_story $STORY --out runs/${TAG}_${COND}_s${STORY} $LORA_T
  python 02_extract_features.py --prepared $PREP --model $MODEL --layers $LAYER \
         --adapter runs/${TAG}_${COND}_s${STORY}/adapter --out features/${TAG}_${COND}_s${STORY}
done
# orthogonalised variant of Idea 1 (recommended if gate said "subtle")
python 04_train_region_misalign.py --prepared $PREP --model $MODEL --condition misalign_md_preserve_lang \
       --orthogonalize --grad_surgery --heldout_story $STORY --out runs/${TAG}_misMD_orth_s${STORY} $LORA_T
python 02_extract_features.py --prepared $PREP --model $MODEL --layers $LAYER \
       --adapter runs/${TAG}_misMD_orth_s${STORY}/adapter --out features/${TAG}_misMD_orth_s${STORY}

# ---------------- Stage C: evaluation ---------------------------------------
python 05_eval_alignment.py --prepared $PREP --heldout_story $STORY --out results/align_${TAG}_s${STORY}.json \
  --features base=features/${TAG}_base/layer${LAYER}.npy \
             ctrl=features/${TAG}_preserve_control_s${STORY}/layer${LAYER}.npy \
             misMD=features/${TAG}_misalign_md_preserve_lang_s${STORY}/layer${LAYER}.npy \
             misMD_orth=features/${TAG}_misMD_orth_s${STORY}/layer${LAYER}.npy \
             misLANG=features/${TAG}_misalign_lang_preserve_md_s${STORY}/layer${LAYER}.npy \
             misALL=features/${TAG}_misalign_all_s${STORY}/layer${LAYER}.npy \
             lmonly=features/${TAG}_lm_only_s${STORY}/layer${LAYER}.npy

if [[ "$MODEL" != *bert* ]]; then
  python 06_eval_linguistic.py --model $MODEL --prepared $PREP --heldout_story $STORY --max_per_task 300 \
         --out results/ling_${TAG}_base.json
  for COND in preserve_control misalign_md_preserve_lang misalign_lang_preserve_md misalign_all lm_only; do
    python 06_eval_linguistic.py --model $MODEL --adapter runs/${TAG}_${COND}_s${STORY}/adapter \
           --prepared $PREP --heldout_story $STORY --max_per_task 300 \
           --out results/ling_${TAG}_${COND}_s${STORY}.json
  done
fi
echo "Done. See results/"
