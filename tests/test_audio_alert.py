import time
import numpy as np
import pytest
from sg_ttrans.risk_engine.adas_controller import ADASLevel
from sg_ttrans.alerts.audio_alert import AudioAlertManager, generate_tone_buffer

def test_generate_tone_buffer():
    # Verify tone generation generates valid audio wave
    sample_rate = 22050
    duration = 0.1  # 100 ms
    freq = 880.0
    waveform = generate_tone_buffer(frequency=freq, duration=duration, sample_rate=sample_rate)
    
    assert isinstance(waveform, np.ndarray)
    assert waveform.dtype == np.float32
    assert len(waveform) == int(sample_rate * duration)
    assert np.max(waveform) <= 1.0
    assert np.min(waveform) >= -1.0
    # Check that amplitude is non-zero
    assert np.max(np.abs(waveform)) > 0.1

def test_alert_audio_level_selection():
    # Manager should only produce audio for Level 2 and Level 3, silent on 0 and 1
    manager = AudioAlertManager(enabled=False)  # enabled=False to avoid opening real hardware in CI/test
    
    sound_0 = manager.get_alert_waveform(ADASLevel.LEVEL_0_NOMINAL)
    sound_1 = manager.get_alert_waveform(ADASLevel.LEVEL_1_VISUAL)
    sound_2 = manager.get_alert_waveform(ADASLevel.LEVEL_2_AUDIO_HAPTIC)
    sound_3 = manager.get_alert_waveform(ADASLevel.LEVEL_3_CRITICAL_AEB)
    
    assert sound_0 is None, "Level 0 must be completely silent"
    assert sound_1 is None, "Level 1 must be completely silent per user preference"
    assert sound_2 is not None, "Level 2 (Audio-Haptic) must have audio waveform"
    assert sound_3 is not None, "Level 3 (Critical AEB) must have audio waveform"
    assert len(sound_2) > 0
    assert len(sound_3) > 0

def test_mute_toggle():
    manager = AudioAlertManager(enabled=True)
    assert not manager.is_muted
    manager.toggle_mute()
    assert manager.is_muted
    manager.toggle_mute()
    assert not manager.is_muted
    manager.stop()

def test_state_updates_and_lifecycle():
    manager = AudioAlertManager(enabled=False)
    assert manager.current_level == ADASLevel.LEVEL_0_NOMINAL
    
    # Update to level 2
    manager.update(ADASLevel.LEVEL_2_AUDIO_HAPTIC)
    assert manager.current_level == ADASLevel.LEVEL_2_AUDIO_HAPTIC
    
    # Update back to nominal
    manager.update(ADASLevel.LEVEL_0_NOMINAL)
    assert manager.current_level == ADASLevel.LEVEL_0_NOMINAL
    
    manager.stop()
    # Idempotent stop
    manager.stop()

def test_rapid_stream_updates_stress():
    # Simulate 60 frames updating the audio manager at high frequency
    manager = AudioAlertManager(enabled=False)
    for i in range(60):
        level = ADASLevel.LEVEL_2_AUDIO_HAPTIC if i % 2 == 0 else ADASLevel.LEVEL_0_NOMINAL
        manager.update(level)
    assert manager.current_level == ADASLevel.LEVEL_0_NOMINAL
    manager.stop()

def test_stream_engine_audio_alert_integration():
    from sg_ttrans.config import SGTransConfig
    from sg_ttrans.inference.stream_engine import StreamEngine
    
    cfg = SGTransConfig(sequence_length=2, d_model=128, num_layers=1)
    engine = StreamEngine(cfg)
    audio_mgr = AudioAlertManager(enabled=False)
    
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    for _ in range(3):
        res, _ = engine.process_frame(dummy_frame, velocity=110.0, ttc=1.0)
        audio_mgr.update(res["adas_level"])
    
    assert audio_mgr.current_level in [
        ADASLevel.LEVEL_0_NOMINAL,
        ADASLevel.LEVEL_1_VISUAL,
        ADASLevel.LEVEL_2_AUDIO_HAPTIC,
        ADASLevel.LEVEL_3_CRITICAL_AEB
    ]
    audio_mgr.stop()

