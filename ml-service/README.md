# ML Training API Service

Automated ML service for training, prediction, and pushing models to Hugging Face.

## Quick Start

### Option 1: Docker (Recommended)

```bash
# Build and run
docker-compose up -d

# View logs
docker-compose logs -f

# Access API docs
# Open browser: http://localhost:8000/docs
```

### Option 2: Docker Build Alternative (if network issues)

```bash
# Use Python 3.11 base image (smaller, faster)
docker build -t ml-service .
docker run -d -p 8000:8000 --name ml-service ml-service

# Or if still having network issues, pull image first:
docker pull python:3.11-slim
docker build -t ml-service .
```

### Option 3: Local (if Docker fails)

```bash
python -m venv venv
venv\Scripts\activate  # Windows
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8000
```

## API Endpoints

- `POST /train` - Train a model
- `POST /evaluate` - Evaluate model
- `POST /predict` - Make predictions  
- `POST /push_to_huggingface` - Push model to HF
- `GET /models` - List trained models

## Usage

```bash
# Train model
curl -X POST http://localhost:8000/train

# Predict
curl -X POST http://localhost:8000/predict

# Push to HuggingFace (set HF_TOKEN in environment)
curl -X POST "http://localhost:8000/push_to_huggingface?model_name=experiment_xxx&hf_repo_name=user/model"
```

## Environment Variables

```bash
# For Hugging Face
export HF_TOKEN=your_token_here  # Linux/Mac
$env:HF_TOKEN="your_token_here"  # Windows PowerShell
```

## Files

- `app.py` - Main API service
- `config.yaml` - Model configuration
- `Dockerfile` - Container definition
- `docker-compose.yml` - Docker orchestration
- `requirements.txt` - Python dependencies

