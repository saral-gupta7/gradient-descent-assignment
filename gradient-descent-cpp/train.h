#pragma once
#include <iostream>
#include <iomanip>
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
    cout << fixed << setprecision(6);
    cout << "\n"
         << setw(7) << "epoch" << "  "
         << setw(10) << "loss" << "  "
         << setw(10) << "m" << "  "
         << setw(10) << "b" << "\n";
    cout << setw(7) << "-------" << "  "
         << setw(10) << "----------" << "  "
         << setw(10) << "----------" << "  "
         << setw(10) << "----------" << "\n";

    for (int e = 1; e <= ep; e++) {
        double gm, gb;              // grads
        grads(mo, xs, ys, gm, gb);  // 1. fwd + 2. loss-grad

        mo.m -= lr * gm; // 3. step against grad
        mo.b -= lr * gb;

        if (e % 500 == 0 || e == 1) {
            double l = mse(mo, xs, ys);
            cout << setw(7) << e << "  "
                 << setw(10) << l << "  "
                 << setw(10) << mo.m << "  "
                 << setw(10) << mo.b << "\n";
        }
    }
}
