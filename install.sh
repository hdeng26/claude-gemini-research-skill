#!/usr/bin/env bash
# Set up the skill's private virtualenv and the config file. Safe to rerun.
set -euo pipefail
cd "$(dirname "$0")"

PY=${PYTHON:-python3}
"$PY" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else "Python 3.10+ required")'

[ -d .venv ] || "$PY" -m venv .venv
.venv/bin/pip install -q --upgrade pip
.venv/bin/pip install -q -r requirements.txt

HOME_DIR=${GEMINI_RESEARCH_HOME:-$HOME/.gemini-research}
mkdir -p "$HOME_DIR"
if [ ! -e "$HOME_DIR/config.env" ]; then
  cp config.env.example "$HOME_DIR/config.env"
  chmod 600 "$HOME_DIR/config.env"
  echo "Created $HOME_DIR/config.env — put your Gemini API key in it."
else
  echo "$HOME_DIR/config.env already exists; left unchanged."
fi
.venv/bin/python -m pytest -q tests | tail -1
