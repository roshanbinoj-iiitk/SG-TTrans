from pc_csg.web.app import build_demo, simulate_counterfactual

def test_simulate_counterfactual():
    result = simulate_counterfactual(
        scenario_name="Scenario A: Nominal Highway",
        mu_friction=0.85,
        max_decel=6.0,
        ego_speed=30.0,
        lead_distance=40.0
    )
    assert "crt_risk" in result
    assert "intervention_level" in result
    assert "pruned_hypotheses_pct" in result
    assert result["plot_image"] is not None
    assert 0.0 <= result["crt_risk"] <= 1.0

def test_build_demo():
    demo = build_demo()
    assert demo is not None
