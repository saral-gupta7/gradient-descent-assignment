# Tunable Parameters — What to Change and What Happens

Only 4 knobs matter in this code. Tune in this order: `lr` → `ep` → init → data size.

---

## 1. `lr` — learning rate (in `main.cpp`)

```cpp
double lr = 0.01;
```

| Value | What happens | Why |
|---|---|---|
| `0.001` (too small) | Loss falls painfully slowly. After 2000 epochs `m` still far from 2. | Steps are tiny; you crawl down the bowl. Fix: raise `lr` or `ep`. |
| `0.01` (good) | Loss falls smoothly to ~0.01, `m→2, b→1`. | Step fits the bowl curvature for `x` in 1..8. |
| `0.1` (too big) | Loss zigzags / explodes to `inf` or `nan`. | You jump over the bowl bottom and bounce higher each step (divergence). |
| `1.0` | Instant `nan`. | Gradient × lr overshoots wildly. |

**Rule:** if loss = `nan/inf` → lower `lr` 10×. If loss barely moves → raise `lr` 3–10×.
**Scale note:** safe `lr` depends on `x` magnitude. If `x` were ~1000 instead of ~8, even `0.01` would diverge. That motivates feature scaling (see §5).

## 2. `ep` — epochs (in `main.cpp`)

```cpp
int ep = 2000;
```

| Value | What happens |
|---|---|
| `10` | Underfit: line still near `m=0, b=0`, loss huge. Model hasn't learned. |
| `500` | Close: `m≈1.9, b≈1.3`, small loss. Fine for a quick demo. |
| `2000` (good) | Converged: `m≈2.0, b≈1.0`, loss ~0.01. |
| `100000` | No real gain; wastes time. On noisy data, can overfit noise (not visible here — line has only 2 params). |

**Rule:** increase `ep` until loss stops falling, then stop. Show the
`ep 1 / 500 / 1000 / 2000` printout — a flattening curve = converged.

## 3. Init — `m, b` start values (in `main.cpp`)

```cpp
Model mo(0.0, 0.0);
```

| Init | What happens |
|---|---|
| `(0, 0)` (used) | Deterministic, converges fine. Best for class demo. |
| `(5, -10)` (far) | Still converges, just takes more epochs. Proves GD is robust. |
| Random large (e.g. ±1000) | Slow start / temporary huge loss, but recovers with small `lr`. |

For this convex (single-bowl) problem, init barely matters — good
talking point: *"unlike deep nets, linear regression has one global
minimum, so any start reaches it."*

## 4. Data — `n` and `x` range (in `data.h`)

- **More points (bigger `n`):** grads average over more data → smoother, more stable steps; each epoch costs more. Batch GD here uses all `n` per step.
- **Bigger `x` values:** gradients contain `e·x`, so large `x` → huge `gm` → need smaller `lr` or normalize `x` first.
- **Noise in `y`:** loss floor rises (can't reach 0); `m, b` jitter around truth. Demonstrates bias–variance: the line can't fit noise, which is correct.

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
| good on train, bad on new `x` | overfit / distribution shift | more varied data, normalize |

**Suggested live demo:** run once with `lr = 0.01` (works), once with
`lr = 0.5` (explodes). Two runs explain the entire parameter story.
