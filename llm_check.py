# llm_check.py
import subprocess
import time
import requests
import os

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
LLM_AVAILABLE = False
USE_DOCKER_MODEL_RUNNER = False

def check_local_ollama():
    """Check if Ollama with llama3.2 is running locally."""
    global LLM_AVAILABLE
    try:
        resp = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=3)
        if resp.status_code == 200 and "llama3.2" in resp.text.lower():
            print("[INFO] Ollama with llama3.2 detected locally.")
            LLM_AVAILABLE = True
            return True
        else:
            print("[INFO] Ollama reachable but llama3.2 not found locally.")
            return False
    except Exception as e:
        print(f"[INFO] Ollama not reachable locally: {e}")
        return False

def start_local_ollama():
    """Try starting local Ollama server."""
    print("[INFO] Starting local Ollama server...")
    subprocess.Popen(
        'ollama serve',
        shell=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )
    time.sleep(10)  # wait a bit for server to start
    return check_local_ollama()

def start_docker_model_runner():
    """Fallback: Use Docker Model Runner for ai/llama3.2"""
    global LLM_AVAILABLE, USE_DOCKER_MODEL_RUNNER
    print("[INFO] Attempting Docker Model Runner for ai/llama3.2...")
    try:
        subprocess.run("docker model pull ai/llama3.2", shell=True, check=True)
        subprocess.run("docker model run ai/llama3.2", shell=True, check=True)
        print("[INFO] Docker Model Runner is running llama3.2")
        LLM_AVAILABLE = True
        USE_DOCKER_MODEL_RUNNER = True
        return True
    except subprocess.CalledProcessError:
        print("[INFO] Docker Model Runner failed.")
        return False

def initialize_llm():
    """Main function to ensure LLM availability."""
    if check_local_ollama():
        return
    if start_local_ollama():
        return
    if start_docker_model_runner():
        return
    print("[INFO] LLM not available. Falling back to standard ABSA/summary.")
