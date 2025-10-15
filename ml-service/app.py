from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
import subprocess
import os
from datetime import datetime
import urllib.request
import json
import shutil
from typing import Optional

app = FastAPI(
    title="ML Training & Prediction Service",
    description="Automated ML service using Ludwig for training, evaluation, and prediction",
    version="1.0.0"
)

DATA_URL = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
CONFIG_FILE = "config.yaml"

# Models for API
class TrainRequest(BaseModel):
    dataset_url: Optional[str] = DATA_URL
    epochs: Optional[int] = 30
    push_to_hf: Optional[bool] = False
    hf_model_name: Optional[str] = None

class PredictRequest(BaseModel):
    use_model: Optional[str] = "latest"


@app.get("/")
def root():
    """Health check endpoint"""
    return {
        "service": "ML Training Service",
        "status": "running",
        "endpoints": {
            "train": "/train",
            "evaluate": "/evaluate",
            "predict": "/predict",
            "models": "/models",
            "push_to_huggingface": "/push_to_huggingface"
        }
    }


@app.get("/models")
def list_models():
    """List all trained models"""
    try:
        if not os.path.exists("results"):
            return {"models": [], "count": 0}
        
        models = sorted(os.listdir("results"), reverse=True)
        model_info = []
        
        for model in models:
            model_path = f"results/{model}"
            if os.path.isdir(model_path):
                model_info.append({
                    "name": model,
                    "path": model_path,
                    "created": os.path.getctime(model_path)
                })
        
        return {"models": model_info, "count": len(model_info)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/train")
def train_model(request: TrainRequest = TrainRequest()):
    """Train a new model with Ludwig"""
    try:
        os.makedirs("data", exist_ok=True)
        os.makedirs("results", exist_ok=True)

        dataset_path = "data/titanic.csv"

        # Download the dataset
        print(f"Downloading dataset from {request.dataset_url}...")
        urllib.request.urlretrieve(request.dataset_url, dataset_path)

        # Unique folder for this training
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"results/experiment_{timestamp}"
        os.makedirs(output_dir, exist_ok=True)

        # Train Ludwig model
        print(f"Training model with {request.epochs} epochs...")
        result = subprocess.run([
            "ludwig", "train",
            "--dataset", dataset_path,
            "--config", CONFIG_FILE,
            "--output_directory", output_dir
        ], check=True, capture_output=True, text=True)

        response = {
            "message": "Model training completed successfully",
            "model_path": output_dir,
            "model_name": f"experiment_{timestamp}",
            "epochs": request.epochs
        }

        # Push to Hugging Face if requested
        if request.push_to_hf and request.hf_model_name:
            hf_result = push_model_to_huggingface(output_dir, request.hf_model_name)
            response["huggingface"] = hf_result

        return response

    except subprocess.CalledProcessError as e:
        raise HTTPException(
            status_code=500,
            detail=f"Training failed: {e.stderr if e.stderr else str(e)}"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@app.post("/evaluate")
def evaluate_model(model_name: Optional[str] = None):
    """Evaluate a trained model"""
    try:
        if not os.path.exists("results"):
            raise HTTPException(status_code=404, detail="No models found")

        # Use specified model or latest
        if model_name:
            model_dir = f"results/{model_name}"
        else:
            models = sorted(os.listdir("results"))
            if not models:
                raise HTTPException(status_code=404, detail="No models found")
            model_dir = f"results/{models[-1]}"

        if not os.path.exists(model_dir):
            raise HTTPException(status_code=404, detail=f"Model {model_name} not found")

        dataset_path = "data/titanic.csv"
        
        # Ludwig creates model in experiment_run/model subdirectory
        if os.path.exists(f"{model_dir}/experiment_run/model"):
            model_path = f"{model_dir}/experiment_run/model"
        else:
            model_path = f"{model_dir}/model"
            
        eval_output = f"{model_dir}/evaluation"

        # Run evaluation
        print(f"Evaluating model at {model_path}...")
        subprocess.run([
            "ludwig", "evaluate",
            "--dataset", dataset_path,
            "--model_path", model_path,
            "--output_directory", eval_output
        ], check=True, capture_output=True, text=True)

        # Read evaluation results
        eval_stats_file = f"{eval_output}/test_statistics.json"
        if os.path.exists(eval_stats_file):
            with open(eval_stats_file, 'r') as f:
                eval_stats = json.load(f)
        else:
            eval_stats = {"message": "Evaluation completed but stats file not found"}

        return {
            "message": "Evaluation completed successfully",
            "model": model_dir,
            "evaluation_results": eval_stats,
            "output_directory": eval_output
        }

    except subprocess.CalledProcessError as e:
        raise HTTPException(
            status_code=500,
            detail=f"Evaluation failed: {e.stderr if e.stderr else str(e)}"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@app.post("/predict")
def predict(request: PredictRequest = PredictRequest()):
    """Make predictions using a trained model"""
    try:
        os.makedirs("predictions", exist_ok=True)
        dataset_path = "data/titanic.csv"

        if not os.path.exists(dataset_path):
            raise HTTPException(status_code=404, detail="Dataset not found. Please train a model first.")

        # Use specified model or latest
        if request.use_model == "latest":
            if not os.path.exists("results"):
                raise HTTPException(status_code=404, detail="No models found")
            models = sorted(os.listdir("results"))
            if not models:
                raise HTTPException(status_code=404, detail="No models found")
            latest_model = models[-1]
        else:
            latest_model = request.use_model

        # Ludwig creates model in experiment_run/model subdirectory
        model_dir = f"results/{latest_model}"
        if os.path.exists(f"{model_dir}/experiment_run/model"):
            model_path = f"{model_dir}/experiment_run/model"
        else:
            model_path = f"{model_dir}/model"
        
        if not os.path.exists(model_path):
            raise HTTPException(status_code=404, detail=f"Model not found at {model_path}")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        pred_output = f"predictions/pred_{timestamp}"

        print(f"Making predictions with model {latest_model}...")
        subprocess.run([
            "ludwig", "predict",
            "--dataset", dataset_path,
            "--model_path", model_path,
            "--output_directory", pred_output
        ], check=True, capture_output=True, text=True)

        return {
            "message": "Prediction completed successfully",
            "model_used": latest_model,
            "output_directory": pred_output,
            "predictions_file": f"{pred_output}/Survived_predictions.csv"
        }

    except subprocess.CalledProcessError as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {e.stderr if e.stderr else str(e)}"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@app.post("/push_to_huggingface")
def push_to_huggingface(model_name: str, hf_repo_name: str, hf_token: Optional[str] = None):
    """Push a trained model to Hugging Face Hub"""
    try:
        model_dir = f"results/{model_name}"
        
        if not os.path.exists(model_dir):
            raise HTTPException(status_code=404, detail=f"Model {model_name} not found")

        # Ludwig creates model in experiment_run/model subdirectory
        if os.path.exists(f"{model_dir}/experiment_run/model"):
            actual_model_path = f"{model_dir}/experiment_run/model"
        else:
            actual_model_path = f"{model_dir}/model"

        # Use huggingface_hub to push
        from huggingface_hub import HfApi, create_repo
        
        # Get token from environment if not provided
        if not hf_token:
            hf_token = os.getenv("HF_TOKEN")
            if not hf_token:
                raise HTTPException(
                    status_code=400,
                    detail="Hugging Face token required. Set HF_TOKEN environment variable or provide in request"
                )

        api = HfApi(token=hf_token)
        
        # Create repo (if it doesn't exist)
        try:
            create_repo(repo_id=hf_repo_name, token=hf_token, exist_ok=True)
        except Exception as e:
            print(f"Repo creation note: {e}")

        # Upload the model directory
        api.upload_folder(
            folder_path=actual_model_path,
            repo_id=hf_repo_name,
            repo_type="model",
            token=hf_token
        )

        return {
            "message": "Model pushed to Hugging Face successfully",
            "repo": f"https://huggingface.co/{hf_repo_name}",
            "model_name": model_name
        }

    except ImportError:
        raise HTTPException(
            status_code=500,
            detail="huggingface_hub not installed. Install it with: pip install huggingface_hub"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error pushing to Hugging Face: {str(e)}")


def push_model_to_huggingface(model_dir: str, hf_repo_name: str):
    """Helper function to push model during training"""
    try:
        hf_token = os.getenv("HF_TOKEN")
        if not hf_token:
            return {"error": "HF_TOKEN not set in environment"}
        
        from huggingface_hub import HfApi, create_repo
        
        # Ludwig creates model in experiment_run/model subdirectory
        if os.path.exists(f"{model_dir}/experiment_run/model"):
            actual_model_path = f"{model_dir}/experiment_run/model"
        else:
            actual_model_path = f"{model_dir}/model"
        
        api = HfApi(token=hf_token)
        create_repo(repo_id=hf_repo_name, token=hf_token, exist_ok=True)
        
        api.upload_folder(
            folder_path=actual_model_path,
            repo_id=hf_repo_name,
            repo_type="model",
            token=hf_token
        )
        
        return {
            "status": "success",
            "repo": f"https://huggingface.co/{hf_repo_name}"
        }
    except Exception as e:
        return {"error": str(e)}
