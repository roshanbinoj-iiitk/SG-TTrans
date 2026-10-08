"""
PC-CSG Interactive Demo — Counterfactual Anomaly Anticipation HUD.

Compatibility wrapper around pc_csg.inference.demo_engine.
"""

from pc_csg.inference.demo_engine import (
    COLORS,
    ScenarioState,
    create_scenarios,
    draw_hud,
    save_scenario_snapshots,
    save_demo_video,
    run_demo,
    main,
)

if __name__ == "__main__":
    main()
