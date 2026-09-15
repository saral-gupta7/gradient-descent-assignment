#include <iostream>
#include <iomanip>
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

    cout << "Gradient Descent (C++) -- fit yh = m*x + b\n";
    cout << "Data:   n = " << xs.size() << ", xs = 1..8, ys ~= 2*x + 1\n";
    cout << "Config: m0 = 0, b0 = 0, lr = " << lr << ", epochs = " << ep << "\n";
    cout << "Start loss: " << fixed << setprecision(6) << mse(mo, xs, ys) << "\n";

    train(mo, xs, ys, lr, ep); // 5. gradient descent

    cout << "\nResult:\n";
    cout << "  final loss : " << mse(mo, xs, ys) << "\n";
    cout << "  final: m = " << mo.m << ", b = " << mo.b << "\n";

    // 6. forward feed on new x
    double xq = 10.0;
    cout.unsetf(ios::floatfield);
    cout << "  fwd(" << xq << ") = " << fixed << setprecision(6)
         << mo.fwd(xq) << " (true ~21)\n";
    return 0;
}
