#pragma once
#include <vector>
#include "model.h"
using namespace std;

// Loss: Mean Squared Error
//   mse = (1/n) * sum((yh - y)^2)
//   yh  = predicted value (model.fwd(x))
//   y   = true value
//   n   = number of points

inline double mse(const Model& mo, const vector<double>& xs,
                  const vector<double>& ys) {
    double s = 0.0; // sum of squared errors
    int n = xs.size();

    for (int i = 0; i < n; i++) {
        double yh = mo.fwd(xs[i]); // forward feed
        double e = yh - ys[i];     // error
        s += e * e;                // squared error
    }
    return s / n;
}

// Grads: dL/dm and dL/db (from calculus, chain rule)
//   dL/dm = (2/n) * sum((yh - y) * x)
//   dL/db = (2/n) * sum((yh - y))
//   gm = grad for m, gb = grad for b

inline void grads(const Model& mo, const vector<double>& xs,
                  const vector<double>& ys, double& gm, double& gb) {
    double sm = 0.0; // acc for m
    double sb = 0.0; // acc for b
    int n = xs.size();

    for (int i = 0; i < n; i++) {
        double yh = mo.fwd(xs[i]); // forward feed
        double e = yh - ys[i];     // error
        sm += e * xs[i];           // x scales m's blame
        sb += e;                   // b shifts all points equally
    }
    gm = (2.0 / n) * sm;
    gb = (2.0 / n) * sb;
}
