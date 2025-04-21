#!/usr/bin/env bash
set -e

# 1) make the venv if it doesn't exist
if [ ! -d "kpler-env" ]; then
  python3 -m venv kpler-env
fi

# 2) install exactly the Kpler SDK (and its pandas<=1.5.3) into that venv
kpler-env/bin/pip install --upgrade pip
kpler-env/bin/pip install kpler.sdk==1.0.53
