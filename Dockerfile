# AI TraceFinder Dockerfile for Render deployment
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Reduce TensorFlow logging and avoid GPU discovery in CPU-only containers
ENV TF_CPP_MIN_LOG_LEVEL=2
ENV CUDA_VISIBLE_DEVICES=""
ENV OMP_NUM_THREADS=2

# Install runtime system dependencies (keep image small)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    ffmpeg \
    ca-certificates \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Optional: install build tools if building packages with C extensions (disabled by default)
ARG INSTALL_BUILD_DEPENDENCIES=0
RUN if [ "$INSTALL_BUILD_DEPENDENCIES" = "1" ]; then \
      apt-get update && apt-get install -y build-essential gcc python3-dev && rm -rf /var/lib/apt/lists/*; \
    fi

# Copy requirements first (for better caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8080/_stcore/health || exit 1

# Run the application
CMD streamlit run app.py --server.port 8080 --server.address 0.0.0.0 --server.headless true