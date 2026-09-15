# Neural Network — Python (1-2-2-1 forward pass)

Implements the "First Pass" slide: input `x = 1`, two hidden layers of
2 neurons, one output neuron, every weight `0.5`, no biases.

`main.py` runs the forward pass twice on the same network:

1. **Linear** — pure weighted sums, the literal reading of the slide
   (`y = 0.5`).
2. **Sigmoid** — logistic applied at each hidden layer, for comparison
   (`y ≈ 0.650778`).

A `sklearn MLPRegressor` (identity activation, same 0.5 weights) cross-checks
the linear result and must print `MATCH`.

## Run (isolated venv, pip only)

```bash
python3 -m venv .venv          # once per repo
source .venv/bin/activate
pip install -r neural-network-python/requirements.txt
python neural-network-python/main.py
```
