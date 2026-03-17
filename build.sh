#!/usr/bin/env bash
set -e

# Install CPU-only PyTorch first to avoid pulling in ~2 GB of CUDA libraries.
# Render free-tier instances are CPU-only, so CUDA is unnecessary.
pip install torch --index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies (torch requirement already satisfied by CPU wheel)
pip install -r requirements.txt
