import sys
import os

print("Verifying imports...")
try:
    from core.config import settings
    print("✅ Settings loaded")
    import core.models
    print("✅ Models loaded")
    import core.llm.client
    print("✅ LLM Client loaded")
    import core.custom_video
    print("✅ Custom Video loaded")
    import core.video_renderer
    print("✅ Video Renderer loaded")
    import main
    print("✅ Main loaded")
except Exception as e:
    import traceback
    traceback.print_exc()
    print(f"❌ Error: {e}")
    sys.exit(1)

print("All modules imported successfully.")
