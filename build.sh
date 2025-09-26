#!/bin/bash

# Install system dependencies
apt-get update
apt-get install -y libglib2.0-0 libsm6 libxext6 libxrender-dev libgomp1 ffmpeg

# Install Python dependencies with specific TensorFlow version
pip install --no-cache-dir -r requirements.txt

# Verify TensorFlow installation
python -c "import tensorflow as tf; print(f'TensorFlow version: {tf.__version__}')"
python -c "import keras; print(f'Keras version: {keras.__version__}')"

#!/bin/bash

echo "🧹 Cleaning up memory and optimizing for deployment..."

# Clean up Python cache
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -name "*.pyc" -delete 2>/dev/null || true

# Remove demo mode components
echo "🎭 Removing demo mode components..."
rm -f demo_detectors.py 2>/dev/null || true

echo "📦 Installing dependencies with memory optimization..."
pip install --no-cache-dir --upgrade pip
pip install --no-cache-dir -r requirements.txt

echo "🔧 Verifying model files..."
python -c "
import os
import tensorflow as tf
print(f'TensorFlow: {tf.__version__}')

model_files = [
    'models/scanner_hybrid.keras',
    'models/hybrid_label_encoder.pkl', 
    'models/hybrid_feat_scaler.pkl',
    'models/scanner_fingerprints.pkl',
    'models/fp_keys.npy',
    'models/tamper_svm_model.pkl',
    'models/tamper_svm_scaler.pkl',
    'models/tamper_svm_threshold.pkl'
]

print('Model files status:')
for file in model_files:
    exists = '✅' if os.path.exists(file) else '❌'
    size = os.path.getsize(file) if os.path.exists(file) else 0
    print(f'{exists} {file} ({size} bytes)')

print('🚀 Build optimization complete!')
"
