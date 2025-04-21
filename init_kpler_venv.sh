#!/usr/bin/env bash
set -e

# only create once
if [ ! -d kpler-env ]; then
  python3 -m venv kpler-env
fi

# install exactly what Kpler needs
./kpler-env/bin/pip install --upgrade pip
./kpler-env/bin/pip install -r kpler_requirements.txt

