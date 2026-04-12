FROM python:3.11-slim

WORKDIR /app

# Install dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY insurance.py inference.py server.py app.py openenv.yaml README.md ./

# Expose FastAPI port
EXPOSE 7860

# Run server via main() entry point
CMD ["python", "server.py"]
