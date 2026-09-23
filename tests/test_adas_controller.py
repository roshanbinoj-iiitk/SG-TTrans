from sg_ttrans.config import SGTransConfig
from sg_ttrans.risk_engine.adas_controller import ADASController, ADASLevel

def test_adas_levels():
    ctrl = ADASController(SGTransConfig())
    # Level 0 (< 0.35)
    act0 = ctrl.evaluate(0.071)
    assert act0.level == ADASLevel.LEVEL_0_NOMINAL
    assert not act0.aeb_active
    assert not act0.lane_hold_active

    # Level 1 (0.35 <= rsi < 0.60)
    act1 = ctrl.evaluate(0.452)
    assert act1.level == ADASLevel.LEVEL_1_VISUAL
    assert not act1.aeb_active

    # Level 2 (0.60 <= rsi < 0.80)
    act2 = ctrl.evaluate(0.650)
    assert act2.level == ADASLevel.LEVEL_2_AUDIO_HAPTIC
    assert not act2.aeb_active

    # Level 3 (>= 0.80)
    act3 = ctrl.evaluate(0.841)
    assert act3.level == ADASLevel.LEVEL_3_CRITICAL_AEB
    assert act3.aeb_active
    assert act3.lane_hold_active
    assert act3.hazard_flashers_active
