#include <iostream>
#include <vector>
#include "model.h"
#include "data.h"
#include "loss.h"
#include "train.h"
using namespace std;

// Flow: data -> model -> train (fwd + grads + update) -> test

int main() {
    vector<double> xs, ys;
    get_data(xs, ys); // 1. load data

    Model mo(0.0, 0.0); // 2. init: m = 0, b = 0

    double lr = 0.01; // 3. step size
    int ep = 2000;    // 4. passes over data

    cout << "start loss: " << mse(mo, xs, ys) << "\n";
    train(mo, xs, ys, lr, ep); // 5. gradient descent
    cout << "final: m = " << mo.m << ", b = " << mo.b << "\n";

    // 6. forward feed on new x
    double xq = 10.0;
    cout << "fwd(" << xq << ") = " << mo.fwd(xq)
              << " (true ~21)\n";
    return 0;
}
