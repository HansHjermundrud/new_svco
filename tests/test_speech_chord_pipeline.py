import unittest

from src.speech_chord_pipeline import SpeechChordRecorder


class SpeechChordRecorderTests(unittest.TestCase):
    def test_records_and_labels_speech_and_chords(self):
        recorder = SpeechChordRecorder()
        recorder.record_speech("play this as a chorus")
        recorder.record_chord("c")
        recorder.record_chord("g")

        self.assertEqual(recorder.transcript(), "play this as a chorus")
        self.assertEqual(recorder.chord_progression(), ["C", "G"])

    def test_builds_claude_payload_for_section_mode(self):
        recorder = SpeechChordRecorder()
        recorder.record_speech("go to the chorus")
        recorder.record_chord("Amin")
        recorder.record_chord("F")

        payload = recorder.to_claude_payload(genre="pop", decade=2010, mode="section", next_section="chorus")

        self.assertEqual(payload["recognized_speech"], "go to the chorus")
        self.assertEqual(payload["recognized_chords"], ["Amin", "F"])
        self.assertEqual(payload["chord_generator_context"]["mode"], "section")
        self.assertEqual(payload["chord_generator_context"]["seed_chords"], "Amin F")
        self.assertEqual(payload["chord_generator_context"]["next_section"], "chorus")
        self.assertEqual(payload["claude_task"]["action"], "route_to_mcp")

    def test_builds_payload_for_generate_and_extend_modes(self):
        recorder = SpeechChordRecorder()
        recorder.record_speech("start from scratch")
        recorder.record_chord("Am")

        generate_payload = recorder.to_claude_payload(genre="pop", decade=2010, mode="generate")
        self.assertNotIn("seed_chords", generate_payload["chord_generator_context"])

        extend_payload = recorder.to_claude_payload(genre="rock", decade=1990, mode="extend")
        self.assertEqual(extend_payload["chord_generator_context"]["seed_chords"], "Am")

    def test_rejects_out_of_range_confidence(self):
        recorder = SpeechChordRecorder()
        with self.assertRaises(ValueError):
            recorder.record_speech("hello", confidence=1.1)
        with self.assertRaises(ValueError):
            recorder.record_chord("C", confidence=-0.1)

    def test_rejects_invalid_chord(self):
        recorder = SpeechChordRecorder()
        with self.assertRaises(ValueError):
            recorder.record_chord("not-a-chord")


if __name__ == "__main__":
    unittest.main()
