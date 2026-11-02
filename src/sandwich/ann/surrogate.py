
# The trained ANN as a black box in physical units: design in, 8 outputs out.
# Loads weights.npz and scalers.json once. After that, predict() can be called
# as many times as the MOGA likes without touching the disk again.

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import CONFIG_DIR
from ..preprocess import RangeScaler
from ..sampling import INPUT_COLS, OUTPUT_COLS
from .network import N_PARAMS, predict as network_predict

ROOT = CONFIG_DIR.parent                                # repository root
DEFAULT_MODEL_DIR = ROOT / "models/ann_baseline"
DEFAULT_SCALERS = ROOT / "data/processed/scalers.json"


# rebuilds a fitted Eq. 16 scaler from one half ("X" or "Y") of scalers.json
def _scaler_from_json(d: dict) -> RangeScaler:
    s = RangeScaler(d["lam1"], d["lam2"])
    s.zmin, s.zmax = np.asarray(d["zmin"], float), np.asarray(d["zmax"], float)
    span = s.zmax - s.zmin
    s.span = np.where(span == 0, 1.0, span)     # same rule as RangeScaler.fit
    return s


class Surrogate:
    """Trained 9-28-27-8 ANN wrapped with the Eq. 16 scalers, used as the MOGA fitness function."""

    # reads the trained weights and the Week 4 scalers, and checks they match the network
    def __init__(self, model_dir: str | Path = DEFAULT_MODEL_DIR,
                 scalers_path: str | Path = DEFAULT_SCALERS):
        self.w = np.load(Path(model_dir) / "weights.npz")["w"]
        if self.w.size != N_PARAMS:
            raise ValueError(f"weights have {self.w.size} values, the 9-28-27-8 network needs {N_PARAMS}")

        with open(scalers_path) as f:
            sc = json.load(f)
        # the scalers must have been written with the same column order the ANN uses
        if sc["input_cols"] != INPUT_COLS or sc["output_cols"] != OUTPUT_COLS:
            raise ValueError(f"{scalers_path} uses a different column order than sampling.py")
        self.xs, self.ys = _scaler_from_json(sc["X"]), _scaler_from_json(sc["Y"])

    # physical inputs (n x 9, INPUT_COLS order) -> physical outputs (n x 8, OUTPUT_COLS order)
    def predict(self, Z: np.ndarray) -> np.ndarray:
        Z = np.atleast_2d(np.asarray(Z, float))
        Xn = self.xs.transform(Z)               # Eq. 16: squash inputs into [0.1, 0.9]
        Yn = network_predict(self.w, Xn)        # forward pass through the network
        return self.ys.inverse_transform(Yn)    # undo Eq. 16 to get kg, unit price, mm

    # same as predict, but takes and returns labelled tables
    def predict_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        Y = self.predict(df[INPUT_COLS].to_numpy(float))
        return pd.DataFrame(Y, columns=OUTPUT_COLS, index=df.index)

    # True for any input outside the range the ANN was trained on
    def outside_training_range(self, Z: np.ndarray) -> np.ndarray:
        Z = np.atleast_2d(np.asarray(Z, float))
        return ((Z < self.xs.zmin) | (Z > self.xs.zmax)).any(axis=1)