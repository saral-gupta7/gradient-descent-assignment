# Gradient Descent from Scratch — Theory + Code Walkthrough

Goal: fit a line `yh = m*x + b` to data, using only raw C++ loops.
No ML libraries. Each file does one job.

```
main.cpp  -> driver (ties everything)
model.h   -> model + forward feed
data.h    -> dataset
loss.h    -> loss (MSE) + gradients
train.h   -> gradient descent loop
```

---

## 1. Theory in 5 steps

### Step 1 — Model (forward feed)
We assume the world is a line:

```
yh = m * x + b
```

- `x` = input, `yh` = prediction ("y-hat")
- `m` = slope (how steep), `b` = bias (where line crosses y-axis)

Forward feed = push `x` through the formula to get `yh`.
That is the entire "neural network" here: one multiply + one add.

### Step 2 — Loss (how wrong are we?)
Mean Squared Error:

```
e   = yh - y        (error per point)
mse = (1/n) * Σ e²  (mean of squared errors)
```

Why square? So positive and negative errors don't cancel, and big
errors are punished more. `mse = 0` means a perfect fit.

### Step 3 — Gradients (which way is downhill?)
Loss is a bowl shape in `(m, b)` space. The gradient = slope of the
bowl = direction of steepest *increase*. So we step the opposite way.

By calculus (chain rule on `((m*x + b) - y)²`):

```
dL/dm = (2/n) * Σ (yh - y) * x
dL/db = (2/n) * Σ (yh - y)
```

Intuition:
- `b` shifts every prediction equally → its blame is just mean error.
- `m` is multiplied by `x` → its blame is error weighted by `x`.
  A point with big `x` pushes `m` harder.

### Step 4 — Update (take one step downhill)
```
m -= lr * gm
b -= lr * gb
```
- `gm, gb` = gradients, `lr` = learning rate (step size).
- Minus sign = walk *downhill*, not uphill.

### Step 5 — Repeat (epochs)
One epoch = one full pass over all data: forward → grads → update.
After hundreds of epochs `m, b` settle at the bowl bottom.

---

## 2. Code walkthrough (file by file, line by line)

### `model.h` — the model + forward feed

```cpp
struct Model {
    double m; // slope
    double b; // bias
```

Theory link: these are the only two numbers we learn.
Everything else (data, loss) exists to tune them.

```cpp
    Model(double m_ = 0.0, double b_ = 0.0) : m(m_), b(b_) {}
```

Constructor with defaults `0, 0`. We start from a flat line
through the origin and let GD move it. Starting at zero keeps the
demo deterministic.

```cpp
    double fwd(double x) const {
        return m * x + b;
    }
```

This IS forward feed: theory Step 1 as one line of code.
Called once per data point per epoch.

### `data.h` — the dataset

```cpp
xs = {1, 2, 3, 4, 5, 6, 7, 8};
ys = {3.1, 5.0, 7.2, 9.1, 11.0, 13.2, 15.1, 17.0};
```

Theory link: ground truth is roughly `y = 2x + 1` plus tiny noise.
So a correct run must converge to `m ≈ 2, b ≈ 1`. If it doesn't,
a hyperparameter is wrong — easy to check in a presentation.

### `loss.h` — `mse()` + `grads()`

`mse()`:

```cpp
double yh = mo.fwd(xs[i]); // forward feed (Step 1)
double e = yh - ys[i];     // error       (Step 2)
s += e * e;                // square + accumulate
...
return s / n;              // mean
```

Each line maps 1:1 to the MSE formula. No hidden library call —
the loop IS the summation `Σ`.

`grads()`:

```cpp
double yh = mo.fwd(xs[i]); // same forward feed
double e = yh - ys[i];     // same error
sm += e * xs[i];           // Σ e*x  -> feeds dL/dm
sb += e;                   // Σ e    -> feeds dL/db
...
gm = (2.0 / n) * sm;
gb = (2.0 / n) * sb;
```

This is Step 3 coded directly. Note `fwd()` is reused — gradients
are computed *from* the forward pass outputs. That reuse is the core
idea of backpropagation in miniature.

### `train.h` — the GD loop

```cpp
for (int e = 1; e <= ep; e++) {
```

One iteration = one epoch (Step 5).

```cpp
    grads(mo, xs, ys, gm, gb);  // fwd + loss-grad
    mo.m -= lr * gm;            // update (Step 4)
    mo.b -= lr * gb;
```

Three lines = the whole algorithm. Order matters: compute grads
at the *current* position, then step. Doing it reversed would use
stale gradients.

```cpp
    if (e % 500 == 0 || e == 1) { ... }
```

Print only snapshots so the console stays readable and you can
show loss decreasing in the presentation.

### `main.cpp` — driver

```cpp
get_data(xs, ys);   // 1. load data
Model mo(0.0, 0.0); // 2. init line at m=0, b=0
double lr = 0.01;   // 3. step size
int ep = 2000;      // 4. passes
train(mo, xs, ys, lr, ep); // 5. run GD
mo.fwd(10.0);       // 6. test forward feed on unseen x
```

Theory link: `fwd(10)` should give `≈ 21` (since `2*10+1 = 21`).
That single number proves generalization: the line works beyond
the training points `1..8`.

---

## 3. Build & run

```bash
g++ -std=c++17 -O2 -Wall main.cpp -o gd
./gd
```

Expected output (loss falls, params converge):

```
start loss: ~100+
ep 1    | loss ... | m ... | b ...
ep 500  | loss ... | m ~1.9 | b ~1.3
ep 2000 | loss ~0.01 | m ~2.0 | b ~1.0
final: m = 2.0x, b = 1.0x
fwd(10) = ~21
```

---

## 4. One-slide summary (for your presentation)

1. **Fwd:** `yh = m*x + b`
2. **Loss:** `mse = mean((yh - y)²)`
3. **Grads:** `gm = mean(2·e·x)`, `gb = mean(2·e)`
4. **Step:** `m -= lr·gm`, `b -= lr·gb`
5. **Loop** 2000× → line fits data.
