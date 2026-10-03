"""
Real-time Non-blocking Audio Alert Subsystem for SG-TTrans ADAS Controller.

Generates automotive acoustic alerts for ADAS Level 2 (Audio-Haptic Alert)
and Level 3 (Critical AEB / Hazard) without blocking the video capture
or neural inference pipelines. Level 0 and Level 1 remain silent.
"""

import threading
import time
import subprocess
import shutil
import numpy as np
from sg_ttrans.risk_engine.adas_controller import ADASLevel

# Attempt import of sounddevice
try:
    import sounddevice as sd
    _SOUNDDEVICE_AVAILABLE = True
except Exception:
    sd = None
    _SOUNDDEVICE_AVAILABLE = False


def generate_tone_buffer(
    frequency: float,
    duration: float,
    sample_rate: int = 22050,
    volume: float = 0.5,
    attack: float = 0.01,
    release: float = 0.02
) -> np.ndarray:
    """
    Synthesize a single sine tone with ADSR attack/release ramp to prevent clipping clicks.
    """
    n_samples = int(sample_rate * duration)
    t = np.linspace(0, duration, n_samples, endpoint=False, dtype=np.float32)
    tone = np.sin(2.0 * np.pi * frequency * t) * volume

    # Apply envelope
    n_att = min(int(sample_rate * attack), n_samples // 2)
    n_rel = min(int(sample_rate * release), n_samples // 2)

    if n_att > 0:
        tone[:n_att] *= np.linspace(0.0, 1.0, n_att, dtype=np.float32)
    if n_rel > 0:
        tone[-n_rel:] *= np.linspace(1.0, 0.0, n_rel, dtype=np.float32)

    return tone.astype(np.float32)


def generate_level2_alert(sample_rate: int = 22050, volume: float = 0.6) -> np.ndarray:
    """
    Urgent Acoustic Alarm: Distinct dual-beep pulse (880 Hz then 1046 Hz).
    """
    tone1 = generate_tone_buffer(880.0, 0.12, sample_rate, volume)
    gap = np.zeros(int(sample_rate * 0.04), dtype=np.float32)
    tone2 = generate_tone_buffer(1046.5, 0.16, sample_rate, volume)
    return np.concatenate([tone1, gap, tone2])


def generate_level3_alert(sample_rate: int = 22050, volume: float = 0.8) -> np.ndarray:
    """
    Critical AEB Hazard Alarm: Rapid, high-urgency alternating siren pulses.
    """
    tone1 = generate_tone_buffer(1318.5, 0.08, sample_rate, volume)  # E6
    gap = np.zeros(int(sample_rate * 0.02), dtype=np.float32)
    tone2 = generate_tone_buffer(1568.0, 0.08, sample_rate, volume)  # G6
    tone3 = generate_tone_buffer(1760.0, 0.10, sample_rate, volume)  # A6
    return np.concatenate([tone1, gap, tone2, gap, tone3])


class AudioAlertManager:
    """
    Asynchronous, non-blocking acoustic alert controller.
    Runs playback on a dedicated background worker daemon.
    """
    def __init__(
        self,
        enabled: bool = True,
        sample_rate: int = 22050,
        volume: float = 0.6
    ):
        self.enabled = enabled
        self.sample_rate = sample_rate
        self.volume = volume
        self.is_muted = False
        self.current_level = ADASLevel.LEVEL_0_NOMINAL

        # Precompute alert waveforms
        self._sound_l2 = generate_level2_alert(self.sample_rate, self.volume)
        self._sound_l3 = generate_level3_alert(self.sample_rate, self.volume)

        # Thread synchronization
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._worker_thread = None

        if self.enabled:
            self._start_worker()

    def _start_worker(self):
        self._worker_thread = threading.Thread(target=self._run_loop, daemon=True)
        self._worker_thread.start()

    def get_alert_waveform(self, level: ADASLevel | int) -> np.ndarray | None:
        """
        Returns precomputed waveform for the specified level.
        Returns None for Level 0 and Level 1 (silent).
        """
        lvl = int(level)
        if lvl == int(ADASLevel.LEVEL_2_AUDIO_HAPTIC):
            return self._sound_l2
        elif lvl == int(ADASLevel.LEVEL_3_CRITICAL_AEB):
            return self._sound_l3
        return None

    def update(self, level: ADASLevel | int):
        """
        Updates current ADAS intervention level.
        Non-blocking, called on every video frame.
        """
        lvl = ADASLevel(int(level))
        with self._lock:
            self.current_level = lvl

    def toggle_mute(self) -> bool:
        """
        Toggles mute state on/off. Returns the new muted state.
        """
        with self._lock:
            self.is_muted = not self.is_muted
            # Stop any actively playing sound if muted
            if self.is_muted and _SOUNDDEVICE_AVAILABLE:
                try:
                    sd.stop()
                except Exception:
                    pass
            return self.is_muted

    def _play_buffer(self, waveform: np.ndarray):
        """
        Plays audio buffer using sounddevice or fallback player.
        """
        if _SOUNDDEVICE_AVAILABLE:
            try:
                sd.play(waveform, samplerate=self.sample_rate)
                sd.wait()
                return
            except Exception:
                pass

        # Fallback: simple brief sleep if sounddevice fails
        duration = len(waveform) / float(self.sample_rate)
        time.sleep(duration)

    def _run_loop(self):
        """
        Background loop handling cadence and playback for active alerts.
        """
        while not self._stop_event.is_set():
            with self._lock:
                lvl = self.current_level
                muted = self.is_muted
                enabled = self.enabled

            if not enabled or muted:
                time.sleep(0.05)
                continue

            waveform = self.get_alert_waveform(lvl)
            if waveform is not None:
                # Play alert sound
                self._play_buffer(waveform)

                # Cadence interval between repetitions
                interval = 0.55 if lvl == ADASLevel.LEVEL_2_AUDIO_HAPTIC else 0.25
                time.sleep(interval)
            else:
                # Nominal or Level 1: sleep briefly
                time.sleep(0.05)

    def stop(self):
        """
        Cleanly stops background thread and audio devices.
        """
        self._stop_event.set()
        if _SOUNDDEVICE_AVAILABLE:
            try:
                sd.stop()
            except Exception:
                pass
        if self._worker_thread is not None and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=1.0)
