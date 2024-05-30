#include "pch.h"

#include"potential.h"
#include"functional.h"
#include"array.h"

#include<math.h>

static inline float mod2pi(float x) {
    const float a = 1 / (2 * pi);
    float y = x * a;
    return y - std::floor(y);
}

template<size_t capacity>
static inline size_t hashFloat2Pi(const float& x) {
    return round(mod2pi(x) * capacity); 
}

static inline float _sin(const float& x) { return sin(x); }
static inline float _cos(const float& x) { return cos(x); }

static ReaderFunc<float, float, sz1d> FSin() {
    static vector<float> xs = linspace(0, 2 * pi, sz1d);
    static auto f = new ReaderFunc<float, float, sz1d>(_sin, hashFloat2Pi<sz1d>, xs);
    return *f;
}

static ReaderFunc<float, float, sz1d> FCos() {
    static vector<float> xs = linspace(0, 2 * pi, sz1d);
    static auto f = new ReaderFunc<float, float, sz1d>(_cos, hashFloat2Pi<sz1d>, xs);
    return *f;
}

static Matrix2f rotationMatrix(const float& a) {
    Matrix2f u; u << fcos(a), -fsin(a), fsin(a), fcos(a);
    return u;
}

static ReaderFunc<float, Matrix2f, sz1d> FRotation() {
    static vector<float> xs = linspace(0, 2 * pi, sz1d);
    static auto f = new ReaderFunc<float, Matrix2f, sz1d>(rotationMatrix, hashFloat2Pi<sz1d>, xs);
    return *f;
}


float fsin(float x) { 
    static auto _fsin = FSin();
    return _fsin(x); 
}

float fcos(float x) { 
    static auto _fcos = FCos();
    return _fcos(x); 
}

Matrix2f FU(float theta) { 
    static auto _u = FRotation();
    return _u(theta); 
}

/*
    The range of (x,y,t): x = X/(2a) in [0,1), y = Y/(a+b) in [0,1), t = Theta/pi in [0,1)
    if szx == szy == szz, the maximal szx is 1024 for the sake of size_t.
*/

template<int n1, int n2, int n3>
inline size_t hashXyt(const xyt& q) {
    const float a1 = n1 - 1,
                a2 = n2 - 1,
                a3 = n3 - 1;
    size_t i = round(q.x * a1),
           j = round(q.y * a2),
           k = round(q.t * a3);
    return i * (n2 * n3) + j * (n3) + k;
}

template<int n1, int n2, int n3>
inline size_t hashXytFloor(const xyt& q) {
    const float a1 = n1 - 1,
                a2 = n2 - 1,
                a3 = n3 - 1;
    size_t i = size_t(q.x * a1),    // floor
           j = size_t(q.y * a2),
           k = size_t(q.t * a3);
    return i * (n2 * n3) + j * (n3)+k;
}

size_t HashXyt(const xyt& q) {
    return hashXyt<szx, szy, szt>(q);
}

xyt Rod::interpolateGradientSimplex(const xyt& q) {
    /*
        fetch potential values of 4 points:
        (i,j,k), (i+1,j,k), (i,j+1,k), (i,j,k+1)
    */
    size_t ijk = HashXyt(q); // ...
    float 
        v000 = fv->data[ijk],
        v100 = fv->data[1 * szy * szt + ijk],
        v010 = fv->data[1 * szt + ijk],
        v001 = fv->data[1 + ijk];
    /*
        solve the linear equation for (A,B,C,D):
        V(x,y,t) = A(x-x0) + B(y-y0) + C(t-t0) + D
    */
    float
        A = (-v000 + v100) * (szx - 1),
        B = (-v000 + v010) * (szy - 1),
        C = (-v000 + v100) * (szt - 1);
    /*
        the gradient: (A,B,C) is already obtained. (if only cauculate gradient, directly return)
        the value: A(x-x0) + B(y-y0) + C(t-t0) + D
    */
    return { A,B,C };
}

/*
    input: q in [0,1] x [0,1] x [0,1]
*/
float Rod::interpolatePotentialSimplex(const xyt& q) {
    /* 
        fetch potential values of 4 points: 
        (i,j,k), (i+1,j,k), (i,j+1,k), (i,j,k+1)
    */
    size_t ijk = hashXytFloor<szx, szy, szt>(q);
    float
        v000 = fv->data[ijk],
        v100 = fv->data[1 * szy * szt + ijk],
        v010 = fv->data[1 * szt + ijk],
        v001 = fv->data[1 + ijk];
    /*
        solve the linear equation for (A,B,C,D):
        V(x,y,t) = A(x-x0) + B(y-y0) + C(t-t0) + D
    */
    float
        A = (-v000 + v100) * (szx - 1),
        B = (-v000 + v010) * (szy - 1),
        C = (-v000 + v001) * (szt - 1),
        D = v000;
    /*
        the energy: A(x-x0) + B(y-y0) + C(t-t0) + D
        since x0 <- floor'(x), there must be x > x0, y > y0, t > t0
    */
    GradientGenerator<szx, szy, szt> gg = { A,B,C,D };
    return gg.calScalar(q);
}
