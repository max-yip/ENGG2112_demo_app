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
    """Returns a list of available model files in the models directory and subdirectories."""
    if not os.path.exists(models_dir):
        return []

    pattern = os.path.join(models_dir, "**", "*.pt")
    models = glob.glob(pattern, recursive=True)

    # Return relative paths so folders are preserved
    return sorted([os.path.relpath(m, models_dir) for m in models])
