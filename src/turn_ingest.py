from __future__ import annotations

from dataclasses import dataclass
import subprocess
import time
from typing import Iterable, Optional


_INT16_MAX = 32768.0


def _require_numpy():
    try:
        import numpy as np
    except ImportError as exc:
        raise ImportError("numpy is required for audio routing. Install it with: pip install numpy") from exc
    return np


def _require_torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError("torch is required for Silero VAD. Install it with: pip install torch") from exc
    return torch


def _rms_from_bytes(frame_bytes: bytes) -> float:
    np = _require_numpy()
    if not frame_bytes:
        return 0.0
    samples = np.frombuffer(frame_bytes, dtype=np.int16).astype("float32")
    if samples.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(samples**2)) / _INT16_MAX)


class SileroVAD:
    def __init__(self, sample_rate: int = 16000, device: str = "cpu", debug: bool = False) -> None:
        torch = _require_torch()
        try:
            model, _ = torch.hub.load(
                repo_or_dir="snakers4/silero-vad",
                model="silero_vad",
                trust_repo=True,
            )
        except Exception as exc:  # pragma: no cover - network/download issues
            raise RuntimeError(
                "Failed to load Silero VAD from torch.hub. Ensure internet access or a cached model."
            ) from exc
        self._torch = torch
        self._model = model.to(device)
        self._model.eval()
        self._device = device
        self._sample_rate = sample_rate
        self._debug = debug

    def reset(self) -> None:
        if hasattr(self._model, "reset_states"):
            self._model.reset_states()

    def speech_probability(self, frame_bytes: bytes) -> float:
        np = _require_numpy()
        if not frame_bytes:
            return 0.0
        audio = np.frombuffer(frame_bytes, dtype=np.int16).astype("float32") / _INT16_MAX
        if audio.size == 0:
            return 0.0
        tensor = self._torch.from_numpy(audio).to(self._device)
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        with self._torch.no_grad():
            prob = self._model(tensor, self._sample_rate)
        if isinstance(prob, self._torch.Tensor):
            value = float(prob.squeeze().cpu().item())
        else:
            value = float(prob)
        value = max(0.0, min(1.0, value))
        if self._debug:
            print(f"[vad] speech_probability={value:.3f}")
        return value


@dataclass(slots=True)
class TurnRouter:
    sample_rate: int = 16000
    frame_duration_s: float = 0.02
    speech_probability_threshold: float = 0.6
    chord_start_rms_threshold: float = 0.02
    chord_keep_rms_threshold: float = 0.015
    chord_hangover_s: float = 0.35
    min_chord_turn_s: float = 0.4
    quiet_seconds: float = 0.8
    rms_ema_alpha: float = 0.2
    debug: bool = False

    def __post_init__(self) -> None:
        self._speech_buffer = bytearray()
        self._chord_buffer = bytearray()
        self._turn_start_time: Optional[float] = None
        self._last_activity_time: Optional[float] = None
        self._chord_active = False
        self._chord_start_time: Optional[float] = None
        self._chord_end_time: Optional[float] = None
        self._hangover_until: Optional[float] = None
        self._rms_ema: Optional[float] = None

    def route(self, frame_bytes: bytes, speech_probability: float, now: Optional[float] = None) -> None:
        now = time.monotonic() if now is None else now
        rms = _rms_from_bytes(frame_bytes)
        if self._rms_ema is None:
            self._rms_ema = rms
        else:
            self._rms_ema = self.rms_ema_alpha * rms + (1 - self.rms_ema_alpha) * self._rms_ema

        if speech_probability >= self.speech_probability_threshold:
            self._record_speech(frame_bytes, now)
            if self._chord_active:
                self._stop_chord(now)
            if self.debug:
                print(f"[router] speech frame rms={rms:.4f}")
            return

        if self._update_chord_state(now):
            self._record_chord(frame_bytes, now)
            if self.debug:
                print(f"[router] chord frame rms={rms:.4f}")

    def should_flush(self, now: Optional[float] = None, max_turn_seconds: Optional[float] = None) -> bool:
        now = time.monotonic() if now is None else now
        if not self._speech_buffer and not self._chord_buffer:
            return False
        if self._turn_start_time is None:
            return False
        if max_turn_seconds is not None and now - self._turn_start_time >= max_turn_seconds:
            return True
        if self._last_activity_time is None:
            return False
        return now - self._last_activity_time >= self.quiet_seconds

    def flush(self) -> tuple[bytes, bytes]:
        speech_bytes = bytes(self._speech_buffer)
        chord_bytes = bytes(self._chord_buffer)
        chord_duration = 0.0
        if self._chord_start_time is not None and self._chord_end_time is not None:
            chord_duration = self._chord_end_time - self._chord_start_time
        if chord_duration < self.min_chord_turn_s:
            chord_bytes = b""
        if self.debug:
            print(
                f"[router] flush speech={len(speech_bytes)} bytes "
                f"chords={len(chord_bytes)} bytes duration={chord_duration:.2f}s"
            )
        self._reset_state()
        return speech_bytes, chord_bytes

    def _record_speech(self, frame_bytes: bytes, now: float) -> None:
        self._speech_buffer.extend(frame_bytes)
        self._mark_activity(now)

    def _record_chord(self, frame_bytes: bytes, now: float) -> None:
        self._chord_buffer.extend(frame_bytes)
        if self._chord_start_time is None:
            self._chord_start_time = now
        self._chord_end_time = now
        self._mark_activity(now)

    def _mark_activity(self, now: float) -> None:
        self._last_activity_time = now
        if self._turn_start_time is None:
            self._turn_start_time = now

    def _update_chord_state(self, now: float) -> bool:
        rms_ema = self._rms_ema or 0.0
        if not self._chord_active and rms_ema >= self.chord_start_rms_threshold:
            self._chord_active = True
            self._chord_start_time = now
            self._chord_end_time = now
            self._hangover_until = now + self.chord_hangover_s
            if self.debug:
                print("[router] chord start")
            return True

        if self._chord_active:
            if rms_ema >= self.chord_keep_rms_threshold:
                self._hangover_until = now + self.chord_hangover_s
            elif self._hangover_until is not None and now > self._hangover_until:
                self._stop_chord(now)
                return False
            return True
        return False

    def _stop_chord(self, now: float) -> None:
        if self.debug:
            print("[router] chord stop")
        self._chord_active = False
        self._hangover_until = None
        if self._chord_end_time is None:
            self._chord_end_time = now

    def _reset_state(self) -> None:
        self._speech_buffer.clear()
        self._chord_buffer.clear()
        self._turn_start_time = None
        self._last_activity_time = None
        self._chord_active = False
        self._chord_start_time = None
        self._chord_end_time = None
        self._hangover_until = None
        self._rms_ema = None


def capture_pyaudio_frames(
    *,
    sample_rate: int = 16000,
    frame_duration_s: float = 0.02,
    device_index: Optional[int] = None,
) -> Iterable[bytes]:
    try:
        import pyaudio
    except ImportError as exc:
        raise ImportError("pyaudio is required for microphone capture. Install it with: pip install pyaudio") from exc

    frame_samples = max(1, int(sample_rate * frame_duration_s))
    audio = pyaudio.PyAudio()
    stream = audio.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=sample_rate,
        input=True,
        frames_per_buffer=frame_samples,
        input_device_index=device_index,
    )
    try:
        while True:
            data = stream.read(frame_samples, exception_on_overflow=False)
            yield data
    finally:
        stream.stop_stream()
        stream.close()
        audio.terminate()


def capture_parec_frames(
    *,
    sample_rate: int = 16000,
    frame_duration_s: float = 0.02,
    pulse_device: Optional[str] = None,
) -> Iterable[bytes]:
    frame_samples = max(1, int(sample_rate * frame_duration_s))
    frame_bytes = frame_samples * 2
    cmd = [
        "parec",
        "--format=s16le",
        f"--rate={sample_rate}",
        "--channels=1",
    ]
    if pulse_device:
        cmd.append(f"--device={pulse_device}")

    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if process.stdout is None:
        raise RuntimeError("parec failed to start audio capture.")
    try:
        while True:
            data = process.stdout.read(frame_bytes)
            if not data:
                break
            if len(data) < frame_bytes:
                continue
            yield data
    finally:
        process.terminate()
        process.wait(timeout=2)
