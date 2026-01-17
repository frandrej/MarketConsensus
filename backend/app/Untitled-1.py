# test_env_location.py
import os
from pathlib import Path

print(f"Current working directory: {os.getcwd()}")
print(f"Script location: {Path(__file__).parent}")
print(f"Looking for .env at: {os.path.join(os.getcwd(), '.env')}")