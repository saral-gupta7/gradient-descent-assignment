# Gradient Descent — Python (sklearn + numpy mirror)

Same toy problem as `gradient-descent-cpp/`: fit `yh = m*x + b` to
`xs = 1..8`, `ys ~= 2*x + 1`, init `m = 0, b = 0`, `lr = 0.01`, `epochs = 2000`.

`main.py` does two things on the same data:

1. **Manual GD (numpy)** — line-by-line mirror of the C++ loop, so the
   printed `ep / loss / m / b` trace matches the C++ output.
2. **Direct fit (sklearn `LinearRegression`)** — closed-form solution,
   no `lr`/`epochs`. Prints `m, b` and checks both agree within `tol = 1e-2`.

## Run (isolated venv, pip only)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python gradient-descent-python/main.py
```

Or from the repo root, run the comparison of both implementations:

```bash
python3 compare.py
```
