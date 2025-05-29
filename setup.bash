#!/usr/bin/env bash
VOLUME_DIR='/opt/payntdev'

set -e

# Fix GIL issues in Stormpy for Saynt
cd /opt/stormpy
git apply ${VOLUME_DIR}/GILhack.patch
python3 setup.py develop

# Install Payntbind
cd ${VOLUME_DIR}/payntbind
python3 setup.py develop

# RUN
cd ${VOLUME_DIR}
python3 -m pip install scipy==1.15.0
