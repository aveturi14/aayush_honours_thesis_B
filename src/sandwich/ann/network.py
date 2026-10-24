
import json
from pathlib import Path
import numpy as np

LAYERS = [9, 28, 27, 8]

INPUT_COLS = ["rho_c", "t_c", "L1", "L2", "L3", "L4", "L5", "L6", "p"]
OUTPUT_COLS = ["SF_c", "SF_f", "SF_int", "SF_wr", "C_delta", "t_sw", "C_t", "W_t"]

SHAPES = []
for _n_in, _n_out in zip(LAYERS[:-1], LAYERS[1:]):
    SHAPES += [(_n_out, _n_in), (_n_out,)]
SIZES = [int(np.prod(s)) for s in SHAPES]
N_PARAMS = sum(SIZES)

def load_split(npz_path):
    d = np.load(npz_path)
    return {k: d[k] for k in ("X_train", "Y_train", "X_val", "Y_val", "X_test", "Y_test")} 

def unpack(w):
    blocks, k = [], 0
    for shape, size in zip(SHAPES, SIZES):
        blocks.append(w[k:k + size].reshape(shape))
        k += size
    return blocks

def pack(blocks):
    return np.concatenate([b.ravel() for b in blocks])

def forward(w, X):
    W1, b1, W2, b2, W3, b3 = unpack(w)  
    a1 = np.tanh(X @ W1.T + b1)     # hidden layer 1, tansig 
    a2 = np.tanh(a1 @ W2.T + b2)    # hidden layer 2, tansig
    y = a2 @ W3.T + b3      # output layer, purelin
    return y, a1, a2

def predict(w, X):
    return forward(w, X)[0]

def mse(w, X, Y):
    return float(np.mean((predict(w, X) - Y) ** 2))

def r_squared(actual, predicted):
    actual, predicted = np.ravel(actual), np.ravel(predicted)
    return 1.0 - np.sum((actual - predicted) ** 2) / np.sum((actual - actual.mean()) ** 2)

def nguyen_widrow(n_out, n_in, rng, in_min=None, in_max=None):
    beta = 0.7 * n_out ** (1.0 / n_in)
    W = rng.uniform(-1, 1, (n_out, n_in))
    W = beta * W / np.linalg.norm(W, axis=1, keepdims=True) 
    b = beta * np.linspace(-1, 1, n_out) * np.sign(W[:, 0])
    
    if in_min is not None:
        scale = 2.0 / (in_max - in_min)
        shift = 1.0 - in_max * scale
        b = b + W @ shift
        W = W * scale
    return W, b

def init_weights(seed, X_reference):
    rng = np.random.default_rng(seed)
    W1, b1 = nguyen_widrow(28, 9, rng, X_reference.min(axis=0), X_reference.max(axis=0))
    W2, b2 = nguyen_widrow(27, 28, rng)
    W3 = rng.uniform(-1, 1, (8, 27))
    b3 = rng.uniform(-1, 1, 8)
    return pack([W1, b1, W2, b2, W3, b3])

def load_model(model_dir):
    model_dir = Path(model_dir)
    w = np.load(model_dir / "weights.npz")["w"]
    with open(model_dir / "training_meta.json") as f:
        meta = json.load(f)
    return w, meta