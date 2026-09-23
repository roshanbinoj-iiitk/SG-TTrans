import numpy as np
import pytest
from sg_ttrans.geometry.facemesh import FaceMeshExtractor

def test_facemesh_extractor_initialization():
    extractor = FaceMeshExtractor()
    assert extractor is not None

def test_facemesh_extractor_on_black_frame():
    extractor = FaceMeshExtractor()
    black_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    landmarks, success = extractor.extract(black_frame)
    # On black frame with no face, success is False and landmarks is None or zero array
    assert success is False
    assert landmarks is None or landmarks.shape == (68, 3)
