"""
Counterfactual Risk Tensor (CRT) Engine — Novel safety metric.

The CRT is a continuous-valued tensor that captures anticipatory risk
across all agent pairs and all counterfactual perturbation modes.

Unlike Time-to-Collision (reactive), the CRT answers:
"Given all physically plausible 'what-if' scenarios, what is the
maximum anticipated risk, and how much lead time do we have?"

CRT(t) = max_{i ∈ critical} max_{k ∈ modes} [R_ik(t) · V_ik(t)]

where R_ik is the counterfactual risk logit and V_ik is PIKV validity.
"""

from typing import Optional

import torch
import torch.nn as nn

from pc_csg.config import PCCSGConfig
from pc_csg.types import (
    CounterfactualMode,
    CounterfactualRiskTensor,
    InterventionDecision,
    InterventionLevel,
)


class CRTEngine(nn.Module):
    """
    Computes the Counterfactual Risk Tensor from CSGA outputs and PIKV validation.

    The CRT aggregates per-agent, per-counterfactual risk scores while
    filtering out physically impossible scenarios (PIKV validity = 0).
    """

    def __init__(self, config: Optional[PCCSGConfig] = None):
        super().__init__()
        if config is None:
            config = PCCSGConfig()
        self.config = config
        self.risk_decay = config.risk_decay
        self.crt_aggregation = config.crt_aggregation

        # Temporal smoothing state
        self.register_buffer(
            "prev_crt", torch.zeros(1), persistent=False,
        )

    def compute_crt(
        self,
        cf_risk_logits: torch.Tensor,
        pikv_validity: torch.Tensor,
        hazard_weights: torch.Tensor,
        node_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute the Counterfactual Risk Tensor.

        Args:
            cf_risk_logits: (B, N, K) — risk under each counterfactual
            pikv_validity: (B, N, K) — physical validity mask from PIKV
            hazard_weights: (B, N) — intrinsic hazard per agent
            node_mask: (B, N) — valid node mask

        Returns:
            crt_scalar: (B,) — aggregated CRT value ∈ [0, 1]
        """
        B, N, K = cf_risk_logits.shape

        # Apply PIKV filter: only physically valid counterfactuals contribute
        filtered_risk = cf_risk_logits * pikv_validity.float()

        # Weight by hazard importance
        weighted_risk = filtered_risk * hazard_weights.unsqueeze(2)

        # Mask out invalid nodes
        if node_mask is not None:
            weighted_risk = weighted_risk * node_mask.unsqueeze(2).float()

        # Aggregate across counterfactuals and agents
        if self.crt_aggregation == "max":
            per_agent_risk, _ = weighted_risk.max(dim=2)     # (B, N)
            crt_scalar, _ = per_agent_risk.max(dim=1)         # (B,)
        elif self.crt_aggregation == "mean":
            per_agent_risk = weighted_risk.mean(dim=2)
            crt_scalar = per_agent_risk.mean(dim=1)
        else:  # "weighted"
            per_agent_risk = weighted_risk.mean(dim=2)
            if node_mask is not None:
                weights = hazard_weights * node_mask.float()
            else:
                weights = hazard_weights
            weight_sum = weights.sum(dim=1).clamp(min=1e-8)
            crt_scalar = (per_agent_risk * weights).sum(dim=1) / weight_sum

        return crt_scalar.clamp(0.0, 1.0)

    def forward(
        self,
        cf_risk_logits: torch.Tensor,
        pikv_validity: torch.Tensor,
        hazard_weights: torch.Tensor,
        node_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass — compute CRT scalar."""
        crt = self.compute_crt(cf_risk_logits, pikv_validity, hazard_weights, node_mask)

        # Temporal smoothing (leaky integration)
        if self.prev_crt.shape[0] == crt.shape[0]:
            crt = self.risk_decay * self.prev_crt + (1 - self.risk_decay) * crt
        self.prev_crt = crt.detach()

        return crt


class PreemptiveInterventionArbiter:
    """
    Maps CRT scalar to graduated preemptive safety intervention levels.

    Unlike reactive systems (which respond AFTER a hazard is detected),
    this arbiter triggers BEFORE the predicted collision based on
    counterfactual analysis.
    """

    def __init__(self, config: Optional[PCCSGConfig] = None):
        if config is None:
            config = PCCSGConfig()
        self.config = config

    def arbitrate(
        self,
        crt_scalar: float,
        per_agent_risk: torch.Tensor,
        per_cf_risk: torch.Tensor,
        hazard_weights: torch.Tensor,
        worst_case_ttc: float = float("inf"),
    ) -> InterventionDecision:
        """
        Determine the preemptive intervention level.

        Args:
            crt_scalar: Aggregated CRT ∈ [0, 1]
            per_agent_risk: (N,) risk per agent
            per_cf_risk: (N, K) risk per counterfactual
            hazard_weights: (N,) hazard weights
            worst_case_ttc: Minimum TTC across all counterfactuals

        Returns:
            InterventionDecision with level, CRT, and action description.
        """
        c = self.config

        # Determine critical agent and mode
        if per_cf_risk.numel() > 0:
            flat_idx = per_cf_risk.argmax().item()
            crit_agent = flat_idx // per_cf_risk.shape[1]
            crit_mode_idx = flat_idx % per_cf_risk.shape[1]
            crit_mode = CounterfactualMode(min(crit_mode_idx, len(CounterfactualMode) - 1))
        else:
            crit_agent = None
            crit_mode = None

        crt = CounterfactualRiskTensor(
            crt_value=crt_scalar,
            per_agent_risk=per_agent_risk.detach().cpu().numpy(),
            per_cf_risk=per_cf_risk.detach().cpu().numpy(),
            critical_agent_id=crit_agent,
            critical_cf_mode=crit_mode,
            worst_case_ttc=worst_case_ttc,
        )

        # ── Graduated intervention logic ──
        if crt_scalar < c.crt_safe:
            level = InterventionLevel.NOMINAL
            desc = f"Nominal operation. CRT={crt_scalar:.3f}, all counterfactuals safe."
            lead_time = worst_case_ttc

        elif crt_scalar < c.crt_caution:
            level = InterventionLevel.ADVISORY
            mode_name = crit_mode.name if crit_mode else "unknown"
            desc = (
                f"Advisory: Agent {crit_agent} has elevated counterfactual risk "
                f"under '{mode_name}' scenario. CRT={crt_scalar:.3f}."
            )
            lead_time = worst_case_ttc

        elif crt_scalar < c.crt_warning:
            level = InterventionLevel.ALERT
            mode_name = crit_mode.name if crit_mode else "unknown"
            desc = (
                f"ALERT: Counterfactual analysis predicts elevated collision risk. "
                f"Agent {crit_agent}, mode='{mode_name}', CRT={crt_scalar:.3f}."
            )
            lead_time = max(worst_case_ttc - 1.0, 0.0)

        elif crt_scalar < c.crt_critical:
            level = InterventionLevel.PRE_INTERVENTION
            desc = (
                f"PRE-INTERVENTION: Activating pre-braking and lane centering. "
                f"CRT={crt_scalar:.3f}, worst-case TTC={worst_case_ttc:.1f}s."
            )
            lead_time = max(worst_case_ttc - 0.5, 0.0)

        else:
            level = InterventionLevel.EMERGENCY
            desc = (
                f"EMERGENCY: Autonomous Emergency Braking + MRM activated. "
                f"CRT={crt_scalar:.3f}, imminent collision predicted."
            )
            lead_time = max(worst_case_ttc, 0.0)

        return InterventionDecision(
            level=level,
            crt=crt,
            action_description=desc,
            preemptive_lead_time=lead_time,
            target_agent_id=crit_agent,
        )
