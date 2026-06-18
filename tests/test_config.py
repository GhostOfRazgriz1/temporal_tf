from temporal_tf.config import Config, default_config

def test_default_config_shapes():
    cfg = default_config()
    assert cfg.n_tokens == (cfg.image_size // cfg.patch_size) ** 2 == 64
    assert cfg.n_layers >= 2 and cfg.horizon_mode in ("A", "B")

def test_config_is_overridable():
    cfg = Config(d_model=32, n_layers=3)
    assert cfg.d_model == 32 and cfg.n_layers == 3
