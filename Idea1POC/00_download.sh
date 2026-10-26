#!/usr/bin/env bash
# Step 0 -- fetch everything the POC needs.
# Run from the region_misalign/ directory. Needs: python>=3.10, git, ~12 GB disk.
set -euo pipefail

mkdir -p data

echo "== 1/4  Python deps"
pip install -q -r requirements.txt

echo "== 2/4  fMRI: Shain et al. Natural Stories fROI data (LANG + MD networks, 78 participants)"
# Primary: Shain, Blank, van Schijndel, Schuler & Fedorenko (2020, Neuropsychologia) -- 3.9 GB, has README.md
# Secondary (same cohort, 2022 J Neurosci working-memory paper)              -- 6.0 GB
# osfclient downloads the whole project; both are public, no login needed.
pip install -q osfclient
if [ ! -d data/osf_eyp8q ]; then
  osf -p eyp8q clone data/osf_eyp8q
fi
# Unzip any archives in place
find data/osf_eyp8q -name "*.zip" -exec sh -c 'unzip -n -q "$1" -d "$(dirname "$1")"' _ {} \;
# Optional: the 2022 release
# osf -p ah429 clone data/osf_ah429 && find data/osf_ah429 -name "*.zip" -exec sh -c 'unzip -n -q "$1" -d "$(dirname "$1")"' _ {} \;

echo "== 3/4  Natural Stories corpus (texts + word timings), Futrell et al. 2021"
if [ ! -d data/naturalstories ]; then
  git clone --depth 1 https://github.com/languageMIT/naturalstories data/naturalstories
fi

echo "== 4/4  Models (HF cache). gpt2 is open; Llama-3.2-1B needs 'huggingface-cli login' + licence."
python - <<'EOF'
from transformers import AutoTokenizer, AutoModelForCausalLM
for m in ["gpt2"]:
    AutoTokenizer.from_pretrained(m); AutoModelForCausalLM.from_pretrained(m)
    print("cached", m)
EOF
# huggingface-cli download meta-llama/Llama-3.2-1B

echo
echo "Done. Next:"
echo "  python 01_prepare_data.py --inspect --data_dir data/osf_eyp8q"
echo "  (check fROI names / columns printed, then)"
echo "  python 01_prepare_data.py --data_dir data/osf_eyp8q --out data/prepared"
