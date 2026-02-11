import unittest
from core.models import VideoPlan, ScriptSegment, TimelineItem

class TestModels(unittest.TestCase):
    def test_script_segment(self):
        seg = ScriptSegment(
            text_es="Hola mundo",
            image_query="Hello world",
            image_prompt="High quality photo of hello world",
            note="Testing"
        )
        self.assertEqual(seg.text_es, "Hola mundo")
        self.assertEqual(seg.note, "Testing")
        
    def test_video_plan_defaults(self):
        plan = VideoPlan(brief="Test video")
        self.assertEqual(plan.brief, "Test video")
        self.assertEqual(plan.target_seconds, 60)
        self.assertEqual(plan.youtube_title_es, "Video personalizado")
        self.assertEqual(plan.segments, [])
        
    def test_video_plan_with_segments(self):
        seg = ScriptSegment(
            text_es="Intro",
            image_query="Intro",
            image_prompt="Intro",
            note="Start"
        )
        plan = VideoPlan(
            brief="Test",
            segments=[seg]
        )
        self.assertEqual(len(plan.segments), 1)
        self.assertEqual(plan.segments[0].text_es, "Intro")

    def test_timeline_item(self):
        item = TimelineItem(prompt="test", start=0.0, end=5.5)
        self.assertEqual(item.start, 0.0)
        self.assertEqual(item.end, 5.5)

if __name__ == "__main__":
    unittest.main()
