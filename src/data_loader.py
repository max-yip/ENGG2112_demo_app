import json
import os
import glob

def load_experiments(json_path="experiments.json"):
    """Loads experiment metrics from a JSON file."""
    if not os.path.exists(json_path):
        return []
    try:
        with open(json_path, "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {json_path}: {e}")
        return []

def get_available_models(models_dir="models"):
    """Returns a list of available model files in the models directory."""
    if not os.path.exists(models_dir):
        return []
    
    # Looking for .pt files which are typical for YOLO and PyTorch
    pattern = os.path.join(models_dir, "*.pt")
    models = glob.glob(pattern)
    # Return just the filenames
    return sorted([os.path.basename(m) for m in models])
