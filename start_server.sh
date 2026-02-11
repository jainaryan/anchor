#!/bin/bash

# Source conda environment manager
source /Users/aryanjain/miniconda3/etc/profile.d/conda.sh

# Activate the dedicated environment
echo "Activating mindmatenv..."
conda activate mindmatenv

# Run the server
echo "Starting MindMate server..."
python -u web/app.py
