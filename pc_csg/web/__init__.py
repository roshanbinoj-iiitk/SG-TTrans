"""
PC-CSG Web Dashboard Package.

Provides Gradio-based interactive browser dashboard for counterfactual
simulation sandbox, real video dashcam analyzer, and benchmark inspector.
"""

from pc_csg.web.app import build_demo, main

__all__ = ["build_demo", "main"]
