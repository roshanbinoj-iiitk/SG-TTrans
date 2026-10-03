"""
Audio and acoustic alert subsystem for SG-TTrans ADAS controller.
"""

from sg_ttrans.alerts.audio_alert import AudioAlertManager, generate_tone_buffer

__all__ = ["AudioAlertManager", "generate_tone_buffer"]
