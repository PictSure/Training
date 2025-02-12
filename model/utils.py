import os
import requests
import torch

def download_model(model_path, url):
    """Download model weights if not already present."""
    if not os.path.exists(model_path):
        print("Downloading model weights...")
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            with open(model_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print("Download complete.")
        else:
            raise Exception(f"Failed to download model: HTTP {response.status_code}")
    else:
        print("Model weights already exist, skipping download.")