
import time
import numpy as np
import pandas as pd
from .network import LAYERS, N_PARAMS, forward, mse, unpack

TRAINLM = dict(epochs=1000, goal=0.0, max_fail=6, min_grad=1e-7, 
               mu=1e-3, mu_dec=0.1, mu_inc=10.0, mu_max=1e10)

def jacobian(w, X):
    W1, b1, W2, b2, W3, b3 = unpack(w)
    N, K = X.shape[0], LAYERS[-1]
    y, a1, a2 = forward(w, X)
    
    delta2 = W3[None, :, :] * (1.0 - a2 ** 2)[:, None, :]
    delta1 = (delta2 @ W2) * (1.0 - a1 ** 2)[:, None, :]
    
    eye = np.eye(K)
    blocks = [
        delta1[:, :, :, None] * X[:, None, None, :],
        delta1, 
        delta2[:, :, :, None] * a1[:, None, None, :],
        delta2,
        eye[None, :, :, None] * a2[:, None, None, :],
        np.broadcast_to(eye, (N, K, K)),
    ]
    
    J = np.concatenate([b.reshape(N, K, -1) for b in blocks], axis=2)
    return J.reshape(N * K, -1), y

def normal_equations(w, X, Y, chunk=1000):
    JtJ = np.zeros((N_PARAMS, N_PARAMS))
    Jtr = np.zeros(N_PARAMS)
    for s in range(0, len(X), chunk):
        J, y = jacobian(w, X[s:s + chunk])
        r = (y - Y[s:s + chunk]).ravel()
        JtJ += J.T @ J
        Jtr += J.T @ r
    return JtJ, Jtr

def train_lm(w_init, data, settings=TRAINLM, report_every=25):
    s = settings
    X_tr, Y_tr = data["X_train"], data["Y_train"]
    X_va, Y_va = data["X_val"], data["Y_val"]
    X_te, Y_te = data["X_test"], data["Y_test"]
    
    w = w_init.copy()
    mu = s["mu"]
    I = np.eye(N_PARAMS)
    perf = mse(w, X_tr, Y_tr)
    
    best = {"epoch": 0, "val": mse(w, X_va, Y_va), "w": w.copy()}
    history = [{"epoch": 0, "train": perf, "val": best["val"], "test": mse(w, X_te, Y_te), "mu": mu}]
    fails = 0
    stop = "maximum epochs reached"
    t0 = time.time()
    
    for epoch in range(1, s["epochs"] + 1):
        JtJ, Jtr = normal_equations(w, X_tr, Y_tr)
        
        if np.linalg.norm(2.0 * Jtr / Y_tr.size) < s["min_grad"]:
            stop = "minimum gradient reached"
            break
        
        accepted = False
        while mu <= s["mu_max"]:
            try:
                dw = np.linalg.solve(JtJ + mu * I, -Jtr)
            except np.linalg.LinAlgError:
                mu *= s["mu_inc"]
                continue
            perf_try = mse(w + dw, X_tr, Y_tr)
            if perf_try < perf:
                w, perf = w + dw, perf_try
                mu *= s["mu_dec"]
                accepted = True
                break
            mu *= s["mu_inc"]
        if not accepted:
            stop = "mu exceeded mu_max"
            break
        
        val, test = mse(w, X_va, Y_va), mse(w, X_te, Y_te)
        history.append({"epoch": epoch, "train": perf, "val": val, "test": test, "mu": mu})
        
        if val < best["val"]:
            best = {"epoch": epoch, "val": val, "w": w.copy()}
            fails = 0
        elif val > best["val"]:
            fails += 1
        
        if report_every and epoch % report_every == 0:
            print(f"epoch {epoch:4d} train {perf:.3e} val {val:.3e} test {test:.3e} "
                    f"mu {mu:.0e} ({time.time() - t0:.0f} s")
        
        if fails >= s["max_fail"]:
            stop = f"validation stop ({s['max_fail']} failures)"
            break
        if perf <= s["goal"]:
            break
    return best, pd.DataFrame(history), stop, time.time() - t0

        
                
            