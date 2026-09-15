#pragma once
#include <iostream>
#include "model.h"
#include "loss.h"
using namespace std;

// Train: batch Gradient Descent
//   mo = model (m, b get updated in place)
//   xs, ys = data
//   lr = learning rate (step size)
//   ep = epochs (full passes over data)

inline void train(Model& mo, const vector<double>& xs,
                  const vector<double>& ys, double lr, int ep) {
    for (int e = 1; e <= ep; e++) {
        double gm, gb;              // grads
        grads(mo, xs, ys, gm, gb);  // 1. fwd + 2. loss-grad

        mo.m -= lr * gm; // 3. step against grad
        mo.b -= lr * gb;

        if (e % 500 == 0 || e == 1) {
            double l = mse(mo, xs, ys);
            cout << "ep " << e << " | loss " << l
                      << " | m " << mo.m << " | b " << mo.b << "\n";
        }
    }
}
