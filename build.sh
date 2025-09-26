#!/bin/bash

# Install system dependencies
apt-get update
apt-get install -y libglib2.0-0 libsm6 libxext6 libxrender-dev libgomp1 ffmpeg

# Install Python dependencies with specific TensorFlow version
pip install --no-cache-dir -r requirements.txt

# Verify TensorFlow installation
python -c "import tensorflow as tf; print(f'TensorFlow version: {tf.__version__}')"
python -c "import keras; print(f'Keras version: {keras.__version__}')"

# Test model loading
python -c "
try:
    import tensorflow as tf
    model = tf.keras.models.load_model('models/scanner_hybrid.keras', compile=False)
    print('✅ Model loads successfully in build environment')
except Exception as e:
    print(f'⚠️ Model loading issue in build: {e}')
    print('Demo mode will be used')
"
