
import json
import numpy as np
import pandas as pd

from .sampling import INPUT_COLS, OUTPUT_COLS

class RangeScaler:
    def __init__(self, lam1: float = 0.1, lam2: float = 0.9):
        self.lam1, self.lam2 = lam1, lam2
        
    def fit(self, Z: np.ndarray) -> "RangeScaler":
        self.zmin, self.zmax = Z.min(axis=0), Z.max(axis=0)
        span = self.zmax - self.zmin
        self.span = np.where(span == 0, 1.0, span)
        return self
    
    def transform(self, Z: np.ndarray) -> np.ndarray:
        return self.lam1 + (self.lam2 - self.lam1) * (Z - self.zmin) / self.span
    
    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        return self.zmin + (X - self.lam1) * self.span / (self.lam2 - self.lam1)
    
    def to_dict(self) -> dict:
        return {"lam1": self.lam1, "lam2": self.lam2,
                "zmin": self.zmin.tolist(), "zmax": self.zmax.tolist()}

def split_indices(n: int, seed: int, fracs=(0.8, 0.1, 0.1)):
    perm = np.random.default_rng(seed).permutation(n)
    n_tr, n_va = round(fracs[0] * n), round(fracs[1] * n)
    return perm[:n_tr], perm[n_tr:n_tr + n_va], perm[n_tr + n_va:]

def prepare(df: pd.DataFrame, seed: int, out_npz: str, out_json: str, fit_on: str = "train"):
    Z_in = df[INPUT_COLS].to_numpy(float)
    Z_out = df[OUTPUT_COLS].to_numpy(float)
    tr, va, te = split_indices(len(df), seed)
    
    rows = tr if fit_on == "train" else slice(None)
    xs = RangeScaler().fit(Z_in[rows])
    ys = RangeScaler().fit(Z_out[rows])
    
    rows = tr if fit_on == "train" else slice(None)
    xs = RangeScaler().fit(Z_in[rows])
    ys = RangeScaler().fit(Z_out[rows])
    
    np.savez(out_npz, 
            X_train=xs.transform(Z_in[tr]), Y_train=ys.transform(Z_out[tr]),
            X_val=xs.transform(Z_in[va]), Y_val=ys.transform(Z_out[va]),
            X_test=xs.transform(Z_in[te]), Y_test=ys.transform(Z_out[te]),
            idx_train=tr, idx_val=va, idx_test=te)
    
    
    with open(out_json, "w") as f:
        json.dump({"fit_on": fit_on, "split_seed": seed, 
                   "input_cols": INPUT_COLS, "output_cols": OUTPUT_COLS,
                   "X": xs.to_dict(), "Y": ys.to_dict()}, f, indent=2)
        
    return xs, ys
