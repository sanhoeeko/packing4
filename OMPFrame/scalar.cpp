#include "pch.h"
#include "defs.h"
#include "scalar.h"
#include "array.h"

float _hertzianSq(const float& x2) {
    return pow(2 - sqrt(x2), 2.5f);
}

float _hertzianSqDR(const float& x2) {
    float x = sqrt(x2);
    return 2.5 * pow(2 - x, 1.5f) / x;
}

float _screenedCoulombSq(const float& x2) {
    const float v0 = exp(-1.0f);
    if (x2 == 0)return 0;
    float x = sqrt(x2) / 2;
    return exp(-x) / x - v0;
}

float _screenedCoulombSqDR(const float& x2) {
    if (x2 == 0)return 0;
    float x = sqrt(x2) / 2;
    return (exp(-x) * (1 + x) / (x * x * x));
}

template<int capacity>
int hash04(const float& x) {
    return x * (capacity / 4);
}

template<>
int anyHasher<float, _h4>(const float& x) {
    return hash04<sz1d>(x);
}

static LookupFunc<float, float, sz1d, _h4> FHertzianSq() {
    static vector<float> xs = linspace(0, 4, sz1d);
    static auto f = new LookupFunc<float, float, sz1d, _h4>(_hertzianSq, xs);
    return *f;
}

static LookupFunc<float, float, sz1d, _h4> FHertzianSqDR() {
    static vector<float> xs = linspace(0, 4, sz1d);
    static auto f = new LookupFunc<float, float, sz1d, _h4>(_hertzianSqDR, xs);
    return *f;
}

static LookupFunc<float, float, sz1d, _h4> FScreenedCoulombSq() {
    static vector<float> xs = linspace(0, 4, sz1d);
    static auto f = new LookupFunc<float, float, sz1d, _h4>(_screenedCoulombSq, xs);
    return *f;
}

static LookupFunc<float, float, sz1d, _h4> FScreenedCoulombSqDR() {
    static vector<float> xs = linspace(0, 4, sz1d);
    static auto f = new LookupFunc<float, float, sz1d, _h4>(_screenedCoulombSqDR, xs);
    return *f;
}

template<>
float scalarPotential<Hertzian>(float x2) {
    static auto f = FHertzianSq();
    if (x2 >= 4)return 0;
    return f(x2);
}

template<>
float potentialDR<Hertzian>(float x2) {
    static auto f = FHertzianSqDR();
    if (x2 >= 4)return 0;
    return f(x2);
}

template<>
float scalarPotential<ScreenedCoulomb>(float x2) {
    static auto f = FScreenedCoulombSq();
    if (x2 >= 4)return 0;
    return f(x2);
}

template<>
float potentialDR<ScreenedCoulomb>(float x2) {
    static auto f = FScreenedCoulombSqDR();
    if (x2 >= 4)return 0;
    return f(x2);
}