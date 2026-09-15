"""Two-hidden-layer neural network, forward pass ("First Pass").

Architecture from the lecture slide:
    input (1) -> hidden-1 (2 neurons) -> hidden-2 (2 neurons) -> output (1)
    every weight = 0.5, no biases.

Neuron rule:  z = sum(w * a_prev),  a = act(z).
The slide shows no activation function, so the literal reading is
linear (pure weighted sums). Sigmoid is shown alongside for comparison,
since hidden layers in class typically use it. The output neuron is
always linear.

Sklearn cross-check: MLPRegressor with identity activation and the same
0.5 weights must reproduce the linear result.
"""

import warnings

import numpy as np
from sklearn.neural_network import MLPRegressor

X = 1.0  # input shown on the slide
W = 0.5  # every edge weight
TOL = 1e-9


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def forward(x, activation="linear"):
    """One forward pass. Returns (hidden-1, hidden-2, output)."""
    act = sigmoid if activation == "sigmoid" else (lambda z: z)
    h1 = act(np.array([x * W, x * W]))                      # hidden-1
    h2 = act(np.array([(h1[0] + h1[1]) * W] * 2))           # hidden-2
    y = float((h2[0] + h2[1]) * W)                          # output (linear)
    return h1, h2, y


def show(activation):
    h1, h2, y = forward(X, activation)
    print(f"\nForward pass ({activation}):")
    print(f"  input     : x  = {X:.6f}")
    print(f"  hidden-1  : h1 = {h1[0]:.6f}, h2 = {h1[1]:.6f}")
    print(f"  hidden-2  : h3 = {h2[0]:.6f}, h4 = {h2[1]:.6f}")
    print(f"  output    : y  = {y:.6f}")
    return y


def sklearn_check(expected):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        reg = MLPRegressor(hidden_layer_sizes=(2, 2), activation="identity",
                           solver="lbfgs", max_iter=1, random_state=0)
        reg.fit([[0.0], [1.0]], [0.0, 1.0])  # shape init only
    reg.coefs_ = [np.full((1, 2), W), np.full((2, 2), W), np.full((2, 1), W)]
    reg.intercepts_ = [np.zeros(2), np.zeros(2), np.zeros(1)]
    got = float(reg.predict([[X]])[0])
    ok = abs(got - expected) < TOL
    print("\nsklearn (MLPRegressor, identity, same 0.5 weights):")
    print(f"  sklearn y = {got:.6f}, numpy y = {expected:.6f} "
          f"-> {'MATCH' if ok else 'MISMATCH'}")
    return ok


def main():
    print("Neural Network (Python) -- 1-2-2-1 forward pass (\"First Pass\")")
    print(f"Input: x = {X:g}, every weight = {W:g}, no biases")

    y_linear = show("linear")
    show("sigmoid")

    ok = sklearn_check(y_linear)
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
