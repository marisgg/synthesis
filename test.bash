#!/usr/bin/env bash

# Fix GIL issues in Stormpy for Saynt
cd /opt/stormpy
git apply GILhack.patch
python3 setup.py develop

# Install Payntbind
cd /opt/payntdev
python3 setup.py develop

# RUN
python3 -m pip install scipy
python3 entrypoint.py
