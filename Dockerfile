# Frag2Drug — Dockerfile
# Author: Dr. Amudha Kumari Duraisamy
#
# Build:  docker build -t frag2drug .
# Run:    docker run -p 8501:8501 frag2drug

FROM python:3.10-slim

# System dependencies required by RDKit and matchms
RUN apt-get update && apt-get install -y --no-install-recommends \
    libxrender1 \
    libxext6 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy dependency files first (layer caching — reinstall only when changed)
COPY requirements.txt .

# Install RDKit via pip (slim image; conda not available)
RUN pip install --no-cache-dir rdkit-pypi && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app.py .
COPY data/example_query.msp data/example_query.msp

# Results directory structure expected by app.py
# Trained models must be mounted at runtime (too large to bake into image)
# docker run -p 8501:8501 -v /path/to/results:/app/results frag2drug
RUN mkdir -p results/data_processing results/model_training/models results/model_training/results

# Streamlit config — disable telemetry and browser auto-open
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_SERVER_PORT=8501

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["streamlit", "run", "app.py", "--server.address=0.0.0.0"]
