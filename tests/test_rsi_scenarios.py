import pytest
from sg_ttrans.config import SGTransConfig
from sg_ttrans.risk_engine.rsi import compute_rsi

def test_scenario_a_normal_blink():
    cfg = SGTransConfig()
    # Scenario A: tau=0.18s, v=80 km/h, TTC=8.0s, yaw=12 deg, pitch=0 deg, y_fatigue=0.12
    # Result: RSI = 0.50(0.12 * 0.113) + 0.35(0.068) + 0.15(0.267) = 0.007 + 0.024 + 0.040 = 0.071
    rsi = compute_rsi(
        fatigue_prob=0.12,
        tau_persist=0.18,
        velocity=80.0,
        ttc=8.0,
        pitch=0.0,
        yaw=12.0,
        config=cfg
    )
    assert pytest.approx(rsi, abs=0.003) == 0.071

def test_scenario_b_moderate_fatigue_yawn():
    cfg = SGTransConfig()
    # Scenario B: tau=2.2s, v=90 km/h, TTC=3.5s, pitch=18 deg, yaw=0 deg, y_fatigue=0.88
    # Result: RSI = 0.50(0.88 * 0.769) + 0.35(0.154) + 0.15(0.400) = 0.338 + 0.054 + 0.060 = 0.452
    rsi = compute_rsi(
        fatigue_prob=0.88,
        tau_persist=2.2,
        velocity=90.0,
        ttc=3.5,
        pitch=18.0,
        yaw=0.0,
        config=cfg
    )
    assert pytest.approx(rsi, abs=0.003) == 0.452

def test_scenario_c_critical_microsleep():
    cfg = SGTransConfig()
    # Scenario C initial (t=1.8s): tau=1.8s, v=110 km/h, TTC=1.4s, pitch=32 deg, yaw=0 deg, y_fatigue=0.98
    # Result: RSI = 0.50(0.98 * 0.699) + 0.35(0.353) + 0.15(0.711) = 0.342 + 0.124 + 0.107 = 0.573
    rsi_initial = compute_rsi(
        fatigue_prob=0.98,
        tau_persist=1.8,
        velocity=110.0,
        ttc=1.4,
        pitch=32.0,
        yaw=0.0,
        config=cfg
    )
    assert pytest.approx(rsi_initial, abs=0.003) == 0.573

    # Scenario C progression: tau=2.8s, v=110 km/h, TTC=0.9s
    # Result: 0.50*(0.98*0.845) + 0.35*(110/130)*(1/1.9) + 0.15*(32/45) = 0.414 + 0.156 + 0.107 = 0.677
    rsi_progression = compute_rsi(
        fatigue_prob=0.98,
        tau_persist=2.8,
        velocity=110.0,
        ttc=0.9,
        pitch=32.0,
        yaw=0.0,
        config=cfg
    )
    assert pytest.approx(rsi_progression, abs=0.003) == 0.677

    # Scenario C emergency escalation: high velocity & critical headway (v=130 km/h, TTC=0.09s, pitch=32 deg)
    # Result: RSI reaches 0.841 => Level 3 Critical Hazard (AEB actuation)
    rsi_critical = compute_rsi(
        fatigue_prob=0.98,
        tau_persist=2.8,
        velocity=130.0,
        ttc=0.09,
        pitch=32.0,
        yaw=0.0,
        config=cfg
    )
    assert pytest.approx(rsi_critical, abs=0.003) == 0.841

def test_rsi_bounds():
    cfg = SGTransConfig()
    # Extreme zero case
    rsi_zero = compute_rsi(0.0, 0.0, 0.0, 100.0, 0.0, 0.0, cfg)
    assert rsi_zero == 0.0

    # Extreme hazard case
    rsi_max = compute_rsi(1.0, 10.0, 200.0, 0.0, 50.0, 50.0, cfg)
    assert rsi_max <= 1.0
