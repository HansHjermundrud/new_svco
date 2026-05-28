from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import importlib
import json
import re
from typing import Any, Iterable

# Root note + optional accidental + optional quality/extensions + optional slash bass note.
# `sus` and `add` forms require an explicit numeric extension (e.g. sus4, add9).
_CHORD_RE = re.compile(
    r"^[A-G](?:#|b)?(?:(?:(?:m|min|maj|dim|aug)?\d*(?:(?:sus|add)\d+)?)|(?:(?:sus|add)\d+))?(?:/[A-G](?:#|b)?)?$"
)
_CHROMATIC_ROOTS = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]
_INT16_MAX = 32767
_INT16_MIN = -32768
_NORMALIZATION_EPSILON = 1e-9


@dataclass(slots=True)
class SpeechEvent:
    text: str
    timestamp_utc: str
    confidence: float = 1.0


@dataclass(slots=True)
class ChordEvent:
    chord: str
    timestamp_utc: str
    confidence: float = 1.0


@dataclass(slots=True)
class MicrophoneConfig:
    sample_rate: int = 16000
    channels: int = 1
    block_duration: float = 0.1
    silence_duration: float = 0.8
    energy_threshold: float = 0.015
    max_segment_duration: float = 12.0
    start_timeout: float = 15.0


@dataclass(slots=True)
class ChordDetectionConfig:
    target_sample_rate: int = 22050
    hop_length: int = 512
    min_chord_duration: float = 0.25
    similarity_threshold: float = 0.35


def _load_optional_dependency(module: str, install_hint: str) -> Any:
    try:
        return importlib.import_module(module)
    except ImportError as exc:
        raise ImportError(
            f"Missing optional dependency '{module}'. Install it with: pip install {install_hint}"
        ) from exc


def _capture_audio_segment(config: MicrophoneConfig) -> tuple[Any, int]:
    np = _load_optional_dependency("numpy", "numpy")
    sd = _load_optional_dependency("sounddevice", "sounddevice")

    if config.block_duration <= 0:
        raise ValueError("block_duration must be greater than 0")
    if config.sample_rate <= 0:
        raise ValueError("sample_rate must be greater than 0")
    if config.channels <= 0:
        raise ValueError("channels must be greater than 0")

    blocksize = max(1, int(config.sample_rate * config.block_duration))
    max_blocks = int(config.max_segment_duration / config.block_duration) if config.max_segment_duration else None
    start_timeout_blocks = int(config.start_timeout / config.block_duration) if config.start_timeout else None

    blocks: list[Any] = []
    started = False
    silence_blocks = 0
    waited_blocks = 0

    with sd.InputStream(
        samplerate=config.sample_rate,
        channels=config.channels,
        dtype="float32",
        blocksize=blocksize,
    ) as stream:
        while True:
            block, _ = stream.read(blocksize)
            waited_blocks += 1
            block = np.asarray(block)
            rms = float(np.sqrt(np.mean(block**2)))

            if rms >= config.energy_threshold:
                started = True
                silence_blocks = 0
                blocks.append(block)
            elif started:
                silence_blocks += 1
                blocks.append(block)
                if silence_blocks * config.block_duration >= config.silence_duration:
                    break
            elif start_timeout_blocks and waited_blocks >= start_timeout_blocks:
                break

            if max_blocks and len(blocks) >= max_blocks:
                break

    if not blocks:
        raise RuntimeError("No audio detected before timeout.")

    if silence_blocks:
        blocks = blocks[:-silence_blocks]

    audio = np.concatenate(blocks, axis=0).astype("float32")
    if config.channels > 1:
        audio = np.mean(audio, axis=1)
    else:
        audio = audio.reshape(-1)
    return audio, config.sample_rate


def _transcribe_with_vosk(audio: Any, sample_rate: int, model_path: str) -> tuple[str, float]:
    np = _load_optional_dependency("numpy", "numpy")
    vosk = _load_optional_dependency("vosk", "vosk")

    model = vosk.Model(model_path)
    recognizer = vosk.KaldiRecognizer(model, sample_rate)

    audio_int16 = (audio * _INT16_MAX).clip(_INT16_MIN, _INT16_MAX).astype(np.int16)
    recognizer.AcceptWaveform(audio_int16.tobytes())
    try:
        result = json.loads(recognizer.Result() or "{}")
    except json.JSONDecodeError as exc:
        raise RuntimeError("Speech recognition returned invalid JSON output.") from exc
    text = str(result.get("text", "")).strip()

    confidence = 1.0
    words = result.get("result", [])
    if isinstance(words, list) and words:
        confidence = float(sum(word.get("conf", 0.0) for word in words) / len(words))
    confidence = max(0.0, min(1.0, confidence))
    return text, confidence


def _build_chord_templates(np: Any) -> tuple[Any, list[str]]:
    templates = []
    labels = []
    for root_index, root in enumerate(_CHROMATIC_ROOTS):
        major = np.zeros(12, dtype="float32")
        major[[root_index, (root_index + 4) % 12, (root_index + 7) % 12]] = 1.0
        minor = np.zeros(12, dtype="float32")
        minor[[root_index, (root_index + 3) % 12, (root_index + 7) % 12]] = 1.0
        templates.append(major)
        labels.append(root)
        templates.append(minor)
        labels.append(f"{root}m")

    templates = np.stack(templates, axis=0)
    templates = templates / (np.linalg.norm(templates, axis=1, keepdims=True) + _NORMALIZATION_EPSILON)
    return templates, labels


def _collapse_chord_frames(
    labels: Iterable[str | None],
    scores: Iterable[float],
    frame_duration: float,
    min_duration: float,
) -> list[tuple[str, float]]:
    collapsed: list[tuple[str, float]] = []
    current_label: str | None = None
    current_scores: list[float] = []
    current_frames = 0

    for label, score in zip(labels, scores):
        if label != current_label:
            if current_label is not None:
                duration = current_frames * frame_duration
                if duration >= min_duration:
                    avg_score = _average_score(current_scores)
                    collapsed.append((current_label, avg_score))
            current_label = label
            current_scores = [score]
            current_frames = 1
        else:
            current_scores.append(score)
            current_frames += 1

    if current_label is not None:
        duration = current_frames * frame_duration
        if duration >= min_duration:
            avg_score = _average_score(current_scores)
            collapsed.append((current_label, avg_score))

    return collapsed


def _detect_chords(audio: Any, sample_rate: int, config: ChordDetectionConfig) -> list[tuple[str, float]]:
    np = _load_optional_dependency("numpy", "numpy")
    librosa = _load_optional_dependency("librosa", "librosa")

    if audio.size == 0:
        return []

    if sample_rate != config.target_sample_rate:
        audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=config.target_sample_rate)
        sample_rate = config.target_sample_rate

    chroma = librosa.feature.chroma_cqt(y=audio, sr=sample_rate, hop_length=config.hop_length)
    if chroma.size == 0:
        return []

    chroma_norm = chroma / (np.linalg.norm(chroma, axis=0, keepdims=True) + _NORMALIZATION_EPSILON)
    templates, labels = _build_chord_templates(np)
    scores = templates @ chroma_norm

    best_indices = np.argmax(scores, axis=0)
    best_scores = scores[best_indices, np.arange(scores.shape[1])]
    frame_labels: list[str | None] = []
    frame_scores: list[float] = []
    for idx, score in zip(best_indices, best_scores):
        if score >= config.similarity_threshold:
            frame_labels.append(labels[int(idx)])
            frame_scores.append(float(score))
        else:
            frame_labels.append(None)
            frame_scores.append(0.0)

    frame_duration = config.hop_length / float(sample_rate)
    return _collapse_chord_frames(frame_labels, frame_scores, frame_duration, config.min_chord_duration)


def _average_score(scores: Iterable[float]) -> float:
    scores_list = list(scores)
    if not scores_list:
        return 0.0
    avg_score = sum(scores_list) / len(scores_list)
    return max(0.0, min(1.0, avg_score))


@dataclass(slots=True)
class SpeechChordRecorder:
    speech_events: list[SpeechEvent] = field(default_factory=list)
    chord_events: list[ChordEvent] = field(default_factory=list)

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _validate_confidence(confidence: float) -> None:
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be between 0.0 and 1.0")

    @staticmethod
    def _normalize_timestamp(timestamp_utc: str | None) -> str:
        if timestamp_utc is None:
            return SpeechChordRecorder._now()
        normalized = timestamp_utc.replace("Z", "+00:00")
        try:
            datetime.fromisoformat(normalized)
        except ValueError as exc:
            raise ValueError("timestamp_utc must be an ISO-8601 datetime string") from exc
        return normalized

    @staticmethod
    def _normalize_chord(chord: str) -> str:
        token = chord.strip().replace(" ", "")
        if not token:
            raise ValueError("Chord token cannot be empty")
        token = token[0].upper() + token[1:]
        if not _CHORD_RE.match(token):
            raise ValueError(f"Invalid chord token: {chord!r}")
        return token

    def record_speech(self, text: str, confidence: float = 1.0, timestamp_utc: str | None = None) -> SpeechEvent:
        if not text or not text.strip():
            raise ValueError("Speech text cannot be empty")
        self._validate_confidence(confidence)
        event = SpeechEvent(text=text.strip(), timestamp_utc=self._normalize_timestamp(timestamp_utc), confidence=confidence)
        self.speech_events.append(event)
        return event

    def record_chord(self, chord: str, confidence: float = 1.0, timestamp_utc: str | None = None) -> ChordEvent:
        self._validate_confidence(confidence)
        normalized = self._normalize_chord(chord)
        event = ChordEvent(chord=normalized, timestamp_utc=self._normalize_timestamp(timestamp_utc), confidence=confidence)
        self.chord_events.append(event)
        return event

    def record_from_microphone(
        self,
        *,
        speech_model_path: str,
        microphone: MicrophoneConfig | None = None,
        chord_config: ChordDetectionConfig | None = None,
    ) -> tuple[SpeechEvent, list[ChordEvent]]:
        """Capture a speech segment followed by a chord segment with a pause between."""
        mic_config = microphone or MicrophoneConfig()
        chord_config = chord_config or ChordDetectionConfig()

        speech_audio, sample_rate = _capture_audio_segment(mic_config)
        speech_text, speech_confidence = _transcribe_with_vosk(speech_audio, sample_rate, speech_model_path)
        if not speech_text:
            raise RuntimeError("Speech recognition returned empty text.")

        speech_event = self.record_speech(speech_text, confidence=speech_confidence)

        chord_audio, chord_sample_rate = _capture_audio_segment(mic_config)
        detected = _detect_chords(chord_audio, chord_sample_rate, chord_config)
        chord_events = [self.record_chord(label, confidence=confidence) for label, confidence in detected]
        return speech_event, chord_events

    def transcript(self) -> str:
        return " ".join(event.text for event in self.speech_events)

    def chord_progression(self) -> list[str]:
        return [event.chord for event in self.chord_events]

    def to_claude_payload(
        self,
        *,
        genre: str,
        decade: int,
        mode: str = "generate",
        next_section: str | None = None,
    ) -> dict[str, Any]:
        if mode not in {"generate", "extend", "section"}:
            raise ValueError(f"Invalid mode: {mode!r}. Must be one of: generate, extend, section")

        seed = " ".join(self.chord_progression())
        context: dict[str, Any] = {
            "genre": genre,
            "decade": decade,
            "mode": mode,
        }
        if mode in {"extend", "section"}:
            context["seed_chords"] = seed
        if mode == "section":
            if not next_section:
                raise ValueError("next_section is required when mode='section'")
            context["next_section"] = next_section

        return {
            "captured_at_utc": self._now(),
            "recognized_speech": self.transcript(),
            "recognized_chords": self.chord_progression(),
            "chord_generator_context": context,
            "claude_task": {
                "action": "route_to_mcp",
                "description": "Use recognized speech and chords to orchestrate the chord progression generator via MCP.",
            },
        }
