"""
PC-CSG Real Video Inference Execution Script.

Compatibility wrapper around pc_csg.inference.real_video_engine.
"""

from pc_csg.inference.real_video_engine import (
    RealVideoInferenceEngine,
    find_default_video,
    run_video_pipeline,
    main,
)

if __name__ == "__main__":
    main()
