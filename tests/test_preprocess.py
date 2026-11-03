
import numpy as np
from sandwich.preprocess import RangeScaler, split_indices

def test_split_sizes_and_disjoint():
    tr, va, te = split_indices(9000, 43)
    assert (len(tr), len(va), len(te)) == (7200, 900, 900)
    assert len(set(tr) | set(va) | set(te)) == 9000
    
def test_split_reproducible():
    a, b = split_indices(9000, 43), split_indices(9000, 43)
    assert all(np.array_equal(x, y) for x, y in zip(a, b))
    
def test_scaler_range_and_roundtrip():
    Z = np.random.default_rng(0).normal(size=(500, 9)) * np.logspace(-3, 3, 9)
    s = RangeScaler().fit(Z)
    X = s.transform(Z)
    
    assert np.allclose(X.min(0), 0.1) and np.allclose(X.max(0), 0.9)
    assert np.allclose(s.inverse_transform(X), Z)