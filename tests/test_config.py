from sg_ttrans.config import SGTransConfig

def test_default_config():
    cfg = SGTransConfig()
    assert cfg.sequence_length == 60
    assert cfg.fps == 30.0
    assert cfg.d_model == 256
    assert cfg.gamma_init == 0.15
    assert cfg.num_classes == 5
    assert cfg.w1 == 0.50 and cfg.w2 == 0.35 and cfg.w3 == 0.15
    assert cfg.delta1 == 0.35 and cfg.delta2 == 0.60 and cfg.delta3 == 0.80
