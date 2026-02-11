import os
import unittest
from unittest.mock import patch

# Mock .env loading
with patch.dict(os.environ, {"VIDEO_QUALITY": "high", "OLLAMA_URL": "http://test:11434"}):
    from core.config import settings

class TestConfig(unittest.TestCase):
    def test_settings_load(self):
        self.assertEqual(settings.video_quality, "high")
        self.assertEqual(settings.ollama_url, "http://test:11434")
        
    def test_default_values(self):
        # We didn't set this in mock, should be default
        self.assertEqual(settings.custom_min_video_sec, 60)
        self.assertEqual(settings.ollama_text_model_short, "gemma2:9b")

    def test_blocked_hosts(self):
        default_sites = {"freepik.com", "pinterest.com"}
        self.assertTrue(default_sites.issubset(settings.blocked_hosts_set))

if __name__ == "__main__":
    unittest.main()
