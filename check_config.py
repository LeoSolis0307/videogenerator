import sys
import os
sys.path.append(os.getcwd())

# 1. Read .env raw
print("--- .env file content ---")
try:
    with open(".env", "r", encoding="utf-8") as f:
        print(f.read())
except Exception as e:
    print(f"Error reading .env: {e}")
print("--------------------------")

# 2. Check os.environ
print(f"os.environ['OLLAMA_TEXT_MODEL'] = '{os.environ.get('OLLAMA_TEXT_MODEL')}'")

# 3. Check pydantic settings
try:
    from core.config import settings
    print(f"settings.ollama_text_model = '{settings.ollama_text_model}'")
    print(f"settings.ollama_text_model_short = '{settings.ollama_text_model_short}'")
    print(f"settings.ollama_text_model_long = '{settings.ollama_text_model_long}'")
except Exception as e:
    print(f"Error loading settings: {e}")
