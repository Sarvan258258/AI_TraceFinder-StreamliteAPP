#!/bin/bash

# Install system dependencies
apt-get update
apt-get install -y libglib2.0-0 libsm6 libxext6 libxrender-dev libgomp1 ffmpeg

# Install Python dependencies
pip install -r requirements.txt
