#!/usr/bin/env bash
# run this at project root (and add it to .streamlit/run.sh so Streamlit Cloud picks it up)

set -e

# 1) remove old if present
rm -rf kplerenv

# 2) make a fresh venv
python3 -m venv kplerenv

# 3) install just the Kpler stack
kplerenv/bin/pip install -r kpler_requirements.txt

