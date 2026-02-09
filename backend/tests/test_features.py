import numpy as np
from src.training.training_from_ims import features_from_snapshot

def test_feature_extraction():
    fake_signal = np.random.randn(20480, 2)  # 2 channels
    feats, names = features_from_snapshot(fake_signal, fs=20000.0)

    assert feats.ndim == 1
    assert len(feats) == len(names)
    assert not np.isnan(feats).any()