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
    print()
    print(f"{'epoch':>7}  {'loss':>10}  {'m':>10}  {'b':>10}")
    print(f"{'-------':>7}  {'----------':>10}  {'----------':>10}  {'----------':>10}")
    for e in range(1, epochs + 1):
        gm, gb = grads(m, b, xs, ys)
        m -= lr * gm
        b -= lr * gb
        if e == 1 or e % 500 == 0:
            print(f"{e:7d}  {mse(m, b, xs, ys):10.6f}  {m:10.6f}  {b:10.6f}")
    return m, b


# --- Direct solution via sklearn (no manual loop) ---
def sklearn_fit(xs, ys):
    reg = LinearRegression()
    reg.fit(xs.reshape(-1, 1), ys)
    m, b = float(reg.coef_[0]), float(reg.intercept_)
    return m, b


def main():
    xs, ys = get_data()

    print("Gradient Descent (Python) -- fit yh = m*x + b")
    print(f"Data:   n = {len(xs)}, xs = 1..8, ys ~= 2*x + 1")
    print(f"Config: m0 = {INIT_M:g}, b0 = {INIT_B:g}, lr = {LR:g}, epochs = {EPOCHS}")
    print(f"Start loss: {mse(INIT_M, INIT_B, xs, ys):.6f}")

    m, b = train(INIT_M, INIT_B, xs, ys, LR, EPOCHS)

    print("\nResult:")
    print(f"  final loss : {mse(m, b, xs, ys):.6f}")
    print(f"  final: m = {m:.6f}, b = {b:.6f}")

    xq = 10.0
    print(f"  fwd({xq:g}) = {fwd(xq, m, b):.6f} (true ~21)")

    # Compare against sklearn's direct fit on the same data.
    sm, sb = sklearn_fit(xs, ys)
    print("\nsklearn (LinearRegression, closed-form):")
    print(f"  sklearn: m = {sm:.6f}, b = {sb:.6f}")
    print(f"  sklearn fwd({xq:g}) = {fwd(xq, sm, sb):.6f} (true ~21)")

    dm, db = abs(m - sm), abs(b - sb)
    match = dm < TOL and db < TOL
    print(f"\nCheck (tol = {TOL}): |dm| = {dm:.6f}, |db| = {db:.6f} "
          f"-> {'MATCH' if match else 'MISMATCH'}")
    return match


if __name__ == "__main__":
    ok = main()
    raise SystemExit(0 if ok else 1)
