import os
import subprocess
import sys

venv_name = "signs"

subprocess.check_call([sys.executable, "-m", "venv", venv_name])

subprocess.check_call([os.path.join(venv_name, "Scripts", "pip"), "install", 
                       "opencv-python", "numpy", "tensorflow", 
                       "mediapipe"])

print(f"Ambiente virtual '{venv_name}' criado e bibliotecas instaladas.")