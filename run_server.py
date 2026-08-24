import os
import sys

# Ensure src module is in path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import uvicorn

if __name__ == "__main__":
    print("🚀 Starting Multilingual AI Voice Assistant Gateway...")
    print("Listening on: ws://0.0.0.0:8000/v1/audio/stream")
    uvicorn.run("src.gateway.server:app", host="0.0.0.0", port=8000, reload=True)
