"""
Context-Aware Level-3 Takeover Safety Arbitrator State Machine.
"""

from typing import Optional
from sg_ttrans.config import STHGSTConfig
from sg_ttrans.types import ArbitrationDecision, TakeoverAction

class TakeoverArbitrator:
    """Euro NCAP 2026 Level-3 Context-Aware Takeover Safety State Machine."""
    def __init__(self, cfg: Optional[STHGSTConfig] = None):
        self.cfg = cfg or STHGSTConfig()

    def arbitrate(
        self,
        eag: float,
        min_ttc: float,
        top_hazard_idx: Optional[int],
    ) -> ArbitrationDecision:
        # Override Rule: Imminent collision mandates emergency MRM
        if min_ttc <= self.cfg.ttc_imminent_threshold:
            return ArbitrationDecision(
                action=TakeoverAction.MINIMUM_RISK_MANEUVER,
                eag=eag,
                causal_hazard_id=top_hazard_idx,
                action_description="Emergency MRM: Imminent collision window detected",
            )

        if eag < self.cfg.eag_handover_thresh:
            return ArbitrationDecision(
                action=TakeoverAction.SAFE_HANDOVER,
                eag=eag,
                causal_hazard_id=None,
                action_description="Safe Handover: Driver attention aligned with hazards",
            )
        elif eag < self.cfg.eag_cue_thresh:
            return ArbitrationDecision(
                action=TakeoverAction.TARGETED_SPATIAL_CUE,
                eag=eag,
                causal_hazard_id=top_hazard_idx,
                action_description=f"Targeted Spatial HUD Cue: Driver missed causal node {top_hazard_idx}",
            )
        else:
            return ArbitrationDecision(
                action=TakeoverAction.MINIMUM_RISK_MANEUVER,
                eag=eag,
                causal_hazard_id=top_hazard_idx,
                action_description="Emergency MRM: Severe epistemic cognitive misalignment",
            )
