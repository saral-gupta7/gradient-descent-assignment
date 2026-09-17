# Tunable Parameters — What to Change and What Happens

Only 4 knobs matter in this code. Tune in this order: `lr` → `ep` → init → data size.

---

## 1. `lr` — learning rate (in `main.cpp`)

```cpp
double lr = 0.01;
```

| Value | What happens | Why |
|---|---|---|
| `0.001` (slow) | After 2000 epochs, `m≈2.054`, `b≈0.777`, loss ≈ 0.028. | Steps are tiny; raise `lr` or `ep`. |
| `0.01` (good) | Loss falls smoothly to ~0.01, `m→2, b→1`. | Step fits the bowl curvature for `x` in 1..8. |
| `0.1` (too big) | Loss zigzags / explodes to `inf` or `nan`. | You jump over the bowl bottom and bounce higher each step (divergence). |
| `1.0` | Diverges rapidly and eventually overflows to `inf`/`nan`. | Gradient × lr overshoots wildly. |

**Rule:** if loss = `nan/inf` → lower `lr` 10×. If loss barely moves → raise `lr` 3–10×.
**Scale note:** safe `lr` depends on `x` magnitude. If `x` were ~1000 instead of ~8, even `0.01` would diverge. That motivates feature scaling (see §5).

## 2. `ep` — epochs (in `main.cpp`)

```cpp
int ep = 2000;
```

| Value | What happens |
|---|---|
| `10` | `m≈2.119`, `b≈0.406`, loss ≈ 0.106; still far from converged. |
| `500` | Close: `m≈1.9, b≈1.3`, small loss. Fine for a quick demo. |
| `2000` (good) | Converged: `m≈2.0, b≈1.0`, loss ~0.01. |
| `100000` | Almost no gain after convergence; wastes time. With a fixed linear model and fixed data, extra epochs alone do not increase model complexity. |

**Rule:** increase `ep` until loss stops falling, then stop. Show the
`ep 1 / 500 / 1000 / 2000` printout — a flattening curve = converged.

## 3. Init — `m, b` start values (in `main.cpp`)

```cpp
Model mo(0.0, 0.0);
```

| Init | What happens |
|---|---|
| `(0, 0)` (used) | Deterministic, converges fine. Best for class demo. |
| `(5, -10)` (far) | Still converges with a stable learning rate, but may take more epochs. |
| Random large (e.g. ±1000) | Slow start / temporary huge loss, but recovers with small `lr`. |

For this convex (single-bowl) problem, init barely matters — good
talking point: *"unlike deep nets, linear regression has one global
minimum, so any start reaches it."*

## 4. Data — `n` and `x` range (in `data.h`)

- **More points (bigger `n`):** each batch gradient averages more examples and each epoch costs more. The optimum may change with the added data.
- **Bigger `x` values:** gradients contain `e·x`, so large `x` → huge `gm` → need smaller `lr` or normalize `x` first.
- **Noise in `y`:** usually raises the minimum training loss and can shift the best-fit `m, b`. Batch GD on fixed data converges to fixed values; it does not keep jittering.

---

## 5. Bonus knobs (if the panel asks "what's next?")

- **Normalize inputs:** `xn = (x - mean)/std` → lets you use bigger `lr`, converges faster. Standard in real projects.
- **Batch vs SGD:** this code is batch GD (whole dataset per step). Stochastic GD (one point per step) is noisier but faster per step on big data; mini-batch is the usual compromise.
- **Loss choice:** MSE punishes outliers hard (squaring). MAE (`mean|e|`) is robust to outliers but its gradient kinks at 0.
- **Print frequency:** `e % 500` in `train.h` — cosmetic only, doesn't affect math. Lower it (e.g. `% 100`) to draw a nicer loss curve.

---

## 6. Debug cheat-sheet (memorize for Q&A)

| Symptom | Cause | Fix |
|---|---|---|
| `loss = nan / inf` | `lr` too big | `lr /= 10` |
| loss flat & high | `lr` too small or `ep` too few | `lr *= 5` or `ep *= 5` |
| loss zigzags down | `lr` slightly big | lower `lr` a bit |
| `m→2` but `b` off | needs more epochs (`b` converges slower) | raise `ep` |
| good on train, bad on new `x` | data may not represent new inputs | collect representative data; evaluate on held-out examples |

**Suggested live demo:** run once with `lr = 0.01` (works), once with
`lr = 0.5` (explodes). Two runs explain the entire parameter story.
