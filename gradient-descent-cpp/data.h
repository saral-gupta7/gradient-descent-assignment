#pragma once
#include <vector>
using namespace std;

// Data: tiny toy set, y = 2*x + 1 (with small noise)
// xs = inputs, ys = true outputs

inline void get_data(vector<double>& xs, vector<double>& ys) {
    xs = {1, 2, 3, 4, 5, 6, 7, 8};
    ys = {3.1, 5.0, 7.2, 9.1, 11.0, 13.2, 15.1, 17.0};
}
