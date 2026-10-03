#!/usr/bin/env sh
set -eu
cd "$(dirname "$0")"
if [ ! -x .venv/bin/python ]; then
  python3 -m venv .venv
fi
if ! .venv/bin/python -c 'import streamlit, disaster_triage' >/dev/null 2>&1; then
  .venv/bin/python -m pip install -e '.[demo]'
fi
.venv/bin/python -m streamlit run demo.py
