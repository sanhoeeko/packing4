#include "pch.h"
#include "array.h"

vector<float> arange(float start, float stop, float step) {
    int n = int((stop - start) / step);
    n = n > 0 ? n : 0;
    vector<float> res; res.reserve(n);
    for (int i = 0; i < n; i++) {
        res.push_back(start + i * step);
    }
    return res;
}

vector<float> linspace(float start, float stop, int size) {
    float step = (stop - start) / size;
    return arange(start, stop, step);
}

vector<float> linspace_including_endpoint(float start, float stop, int size) {
    float step = (stop - start) / (size - 1);
    return arange(start, stop + step, step);
}

static_array<xyt> meshgrid3(vector<float>& xs, vector<float>& ys, vector<float>& ts) {
    size_t n1 = xs.size(), n2 = ys.size(), n3 = ts.size();
    static_array<xyt> res(n1 * n2 * n3);
    xyt* ptr = res.data;
    for (float x : xs) for (float y : ys) for (float t : ts) *ptr++ = { x,y,t };
    return res;
}