#pragma once
using namespace std;

// Model: y = m * x + b
// m = slope, b = bias (intercept)
// fwd() = forward feed: input x -> output yh

struct Model {
    double m; // slope
    double b; // bias

    Model(double m_ = 0.0, double b_ = 0.0) : m(m_), b(b_) {}

    // Forward feed: predict y from x
    double fwd(double x) const {
        return m * x + b;
    }
};
