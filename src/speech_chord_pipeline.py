from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import re
from typing import Any

# Root note + optional accidental + optional quality/extensions + optional slash bass note.
# `sus` and `add` forms require an explicit numeric extension (e.g. sus4, add9).
_CHORD_RE = re.compile(
    r"^[A-G](?:#|b)?(?:(?:(?:m|min|maj|dim|aug)?\d*(?:(?:sus|add)\d+)?)|(?:(?:sus|add)\d+))?(?:/[A-G](?:#|b)?)?$"
)


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
