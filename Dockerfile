FROM python:3.11-slim

WORKDIR /app

# Install dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY insurance.py inference.py server.py openenv.yaml ./

# Expose FastAPI port
EXPOSE 8000

# Default: run the FastAPI server
# Override CMD to run inference: docker run ... python inference.py
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "7860"]
