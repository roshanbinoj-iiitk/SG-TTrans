"""
Global configuration dataclass for SG-TTrans framework.
"""

from dataclasses import dataclass

@dataclass
class SGTransConfig:
    # Temporal sequence
    sequence_length: int = 60       # T = 60 frames
    fps: float = 30.0               # 30 FPS => 2.0s duration
    
    # Feature Dimensions
    d_vis: int = 256                # Spatial appearance feature token dimension
    d_geom: int = 16                # Geometric kinematics token dimension
    d_model: int = 256              # Unified transformer model dimension
    
    # Transformer Architecture
    num_heads: int = 4              # Multi-Head Attention heads (d_k = 64)
    num_layers: int = 4             # Number of TDDA Transformer blocks
    d_ff: int = 1024                # Feed-forward hidden dimension
    gamma_init: float = 0.15        # Learnable TDDA distance decay initialization
    
    # Classification Head
    num_classes: int = 5            # 0: Alert, 1: Drowsy, 2: Microsleep, 3: Yawn, 4: Distracted
    focal_gamma: float = 2.0        # Multi-class Focal Loss focusing parameter
    
    # Geometric Thresholds
    delta_ear: float = 0.20         # Eye closure threshold for PERCLOS
    delta_mar: float = 0.50         # Mouth opening threshold for yawning
    perclos_window: int = 60        # Sliding window for PERCLOS (frames)
    
    # Vehicular Dynamics & Risk-Aware Safety Index (RSI)
    tau0: float = 1.5               # Fatigue persistence saturation time constant (seconds)
    v_max: float = 130.0            # Highway maximum velocity calibration (km/h)
    w1: float = 0.50                # Weight: fatigue persistence
    w2: float = 0.35                # Weight: vehicle speed & collision headway (TTC)
    w3: float = 0.15                # Weight: head pose deviation
    
    # Graduated ADAS Intervention Thresholds
    delta1: float = 0.35            # Level 1: Visual Warning Prompt + Gentle Chime
    delta2: float = 0.60            # Level 2: Audio-Haptic Alert
    delta3: float = 0.80            # Level 3: Emergency Braking + Lane Centering Active Hold
