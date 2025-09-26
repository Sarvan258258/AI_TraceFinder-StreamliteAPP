#!/bin/bash

# Set default port if not provided
PORT=${PORT:-8080}

echo "🚀 Starting AI TraceFinder on port $PORT"
echo "📊 Python version: $(python --version)"
echo "🔧 TensorFlow version: $(python -c 'import tensorflow as tf; print(tf.__version__)' 2>/dev/null || echo 'Not available')"

# Start the Streamlit application
streamlit run app.py \
  --server.port=$PORT \
  --server.address=0.0.0.0 \
  --server.headless=true \
  --server.enableCORS=false \
  --server.enableXsrfProtection=false
