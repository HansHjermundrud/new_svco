# new_svco

Speech and chord capture pipeline that labels recognized input and sends a Claude/MCP-ready payload.

## What is implemented

- Record and label speech events.
- Record and normalize chord events.
- Build a structured payload that includes:
  - recognized speech,
  - recognized chord sequence,
  - chord-generation context (`genre`, `decade`, mode: `generate|extend|section`),
  - Claude task metadata for MCP routing.

## Usage

```python
from src.speech_chord_pipeline import SpeechChordRecorder

recorder = SpeechChordRecorder()
recorder.record_speech("make this a chorus")
recorder.record_chord("C")
recorder.record_chord("G")
recorder.record_chord("Amin")
recorder.record_chord("F")

payload = recorder.to_claude_payload(
    genre="pop",
    decade=2010,
    mode="section",
    next_section="chorus",
)
```

`payload` is the handoff object for Claude to route via MCP to the rest of the system.

## Microphone capture

Install dependencies:

```bash
pip install -r requirements.txt
```

Download a Vosk model and provide the local path (for example, `vosk-model-small-en-us-0.15`).

```python
from src.speech_chord_pipeline import SpeechChordRecorder

recorder = SpeechChordRecorder()
speech_event, chord_events = recorder.record_from_microphone(
    speech_model_path="/path/to/vosk-model-small-en-us-0.15",
)
```

## Run tests

```bash
python -m unittest discover -s tests -p "test_*.py"
```
