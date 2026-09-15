"""Gradient descent from scratch (numpy) + direct solution (sklearn).

Mirrors gradient-descent-cpp/ exactly:
  data:  xs = 1..8, ys ~= 2*x + 1 (same tiny noisy set)
  model: yh = m*x + b, init m = 0, b = 0
  train: batch GD, lr = 0.01, epochs = 2000

Then fits the same data with sklearn's LinearRegression (closed-form,
no lr/epochs) and checks both agree within tolerance.
"""

import numpy as np
from sklearn.linear_model import LinearRegression

# --- Config (same knobs as main.cpp) ---
LR = 0.01
EPOCHS = 2000
INIT_M = 0.0
INIT_B = 0.0
TOL = 1e-2  # C++ GD vs sklearn agreement tolerance


# --- Data (same as data.h) ---
def get_data():
    xs = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=float)
    ys = np.array([3.1, 5.0, 7.2, 9.1, 11.0, 13.2, 15.1, 17.0], dtype=float)
    return xs, ys


# --- Model + loss (same as model.h / loss.h) ---
def fwd(x, m, b):
    return m * x + b


def mse(m, b, xs, ys):
    return float(np.mean((fwd(xs, m, b) - ys) ** 2))


def grads(m, b, xs, ys):
    e = fwd(xs, m, b) - ys
    gm = float(np.mean(2 * e * xs))
    gb = float(np.mean(2 * e))
    return gm, gb


# --- Train: batch gradient descent (same as train.h) ---
def train(m, b, xs, ys, lr=LR, epochs=EPOCHS):
    for e in range(1, epochs + 1):
        gm, gb = grads(m, b, xs, ys)
        m -= lr * gm
        b -= lr * gb
        if e == 1 or e % 500 == 0:
            print(f"ep {e} | loss {mse(m, b, xs, ys)} | m {m} | b {b}")
    return m, b


# --- Direct solution via sklearn (no manual loop) ---
def sklearn_fit(xs, ys):
    reg = LinearRegression()
    reg.fit(xs.reshape(-1, 1), ys)
    m, b = float(reg.coef_[0]), float(reg.intercept_)
    return m, b


def main():
    xs, ys = get_data()

    print(f"start loss: {mse(INIT_M, INIT_B, xs, ys)}")
    m, b = train(INIT_M, INIT_B, xs, ys, LR, EPOCHS)
    print(f"final: m = {m}, b = {b}")

    xq = 10.0
    print(f"fwd({xq}) = {fwd(xq, m, b)} (true ~21)")

    # Compare against sklearn's direct fit on the same data.
    sm, sb = sklearn_fit(xs, ys)
    print(f"sklearn: m = {sm}, b = {sb}")
    print(f"sklearn fwd({xq}) = {fwd(xq, sm, sb)} (true ~21)")

    dm, db = abs(m - sm), abs(b - sb)
    match = dm < TOL and db < TOL
    print(f"match within tol={TOL}: {match} (|dm|={dm:.5f}, |db|={db:.5f})")
    return match


if __name__ == "__main__":
    ok = main()
    raise SystemExit(0 if ok else 1)
