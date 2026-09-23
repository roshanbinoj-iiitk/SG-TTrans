"""
Graduated ADAS Safety Controller mapping RSI(t) to progressive vehicle interventions.
"""

from enum import IntEnum
from dataclasses import dataclass
from sg_ttrans.config import SGTransConfig

class ADASLevel(IntEnum):
    LEVEL_0_NOMINAL = 0
    LEVEL_1_VISUAL = 1
    LEVEL_2_AUDIO_HAPTIC = 2
    LEVEL_3_CRITICAL_AEB = 3

@dataclass
class ADASAction:
    level: ADASLevel
    name: str
    description: str
    aeb_active: bool
    lane_hold_active: bool
    hazard_flashers_active: bool

class ADASController:
    """
    Evaluates dynamic Risk-Aware Safety Index against thresholds:
        - delta1 = 0.35: Visual warning
        - delta2 = 0.60: Audio-haptic alert
        - delta3 = 0.80: Emergency braking & lane centering hold
    """
    def __init__(self, config: SGTransConfig | None = None):
        self.cfg = config if config is not None else SGTransConfig()

    def evaluate(self, rsi: float) -> ADASAction:
        if rsi >= self.cfg.delta3:
            return ADASAction(
                level=ADASLevel.LEVEL_3_CRITICAL_AEB,
                name="Level 3: Critical Hazard",
                description="Emergency Braking (AEB) Assist, Lane Centering Active Hold, Autonomous Hazard Flasher",
                aeb_active=True,
                lane_hold_active=True,
                hazard_flashers_active=True,
            )
        elif rsi >= self.cfg.delta2:
            return ADASAction(
                level=ADASLevel.LEVEL_2_AUDIO_HAPTIC,
                name="Level 2: Audio-Haptic Alert",
                description="Urgent Acoustic Alarm + Steering Wheel Haptic Pulse",
                aeb_active=False,
                lane_hold_active=False,
                hazard_flashers_active=False,
            )
        elif rsi >= self.cfg.delta1:
            return ADASAction(
                level=ADASLevel.LEVEL_1_VISUAL,
                name="Level 1: Visual Warning",
                description="Visual Dashboard Prompt + Gentle Chime",
                aeb_active=False,
                lane_hold_active=False,
                hazard_flashers_active=False,
            )
        else:
            return ADASAction(
                level=ADASLevel.LEVEL_0_NOMINAL,
                name="Level 0: Nominal Operation",
                description="Safe driving state, no intervention",
                aeb_active=False,
                lane_hold_active=False,
                hazard_flashers_active=False,
            )
