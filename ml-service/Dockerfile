# Use official Ludwig image - already has everything installed!
FROM ludwigai/ludwig:latest

WORKDIR /app

# Install only the missing packages (FastAPI + HuggingFace)
RUN pip install --no-cache-dir fastapi uvicorn[standard] huggingface_hub

# Copy application files
COPY app.py config.yaml ./

# Create directories
RUN mkdir -p data results predictions

EXPOSE 8000

# Override the Ludwig entrypoint
ENTRYPOINT []

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
