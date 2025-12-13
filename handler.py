"""
RunPod Serverless Handler
Entry point for RunPod Serverless deployment
"""

import runpod

from api import handler

# Preload model at cold start
print("Preloading model for RunPod Serverless...")
from core import load_model

load_model()
print("Model preloaded!")

# Register handler with RunPod
runpod.serverless.start({"handler": handler})
