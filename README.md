# Gradient Descent — C++ vs Python

Toy linear regression (`yh = m*x + b` on `xs = 1..8`, `ys ~= 2*x + 1`),
implemented twice on identical data + hyperparameters
(`m = b = 0`, `lr = 0.01`, `epochs = 2000`).

| Folder | What it is |
|---|---|
| `gradient-descent-cpp/` | Original from-scratch C++ (loops only, no ML lib). `make run` to run. See `THEORY.md` / `PARAMS.md`. |
| `gradient-descent-python/` | Same GD mirrored with numpy **plus** a direct `sklearn.LinearRegression` fit. Pip + isolated `.venv`. |

## Quick start

```bash
# C++
make -C gradient-descent-cpp run

# Python (isolated venv)
python3 -m venv .venv
source .venv/bin/activate
pip install -r gradient-descent-python/requirements.txt
python gradient-descent-python/main.py

# Compare both (builds C++, runs Python, diffs m/b)
python compare.py
```

## Result

Both converge to `m ≈ 2.0, b ≈ 1.1, fwd(10) ≈ 21.07` and agree within
`tol = 1e-2` (see `compare.py` output). The sklearn closed-form fit lands
on the same line, confirming the hand-written GD.
