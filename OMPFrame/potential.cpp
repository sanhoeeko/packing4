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
    return round(mod2pi(x) * capacity);  // signed before mudolo
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

Gate::Gate()
{
}

Gate::Gate(float a, float b) : a(a), b(b) { ; }


static inline float modpi(float x) {
    const float a = 1 / pi;
    float y = x * a;
    return y - std::floor(y);
}

xyt Gate::transform(const xyt& q)
{
    return
    {
        abs(q.x) / (2 * a),
        abs(q.y) / (a + b),
        (q.x > 0) ^ (q.y > 0) ? 1 - modpi(q.t) : modpi(q.t)
    };
}

xyt Gate::inverse(const xyt& q)
{
    return
    {
        (2 * a) * q.x,
        (a + b) * q.y,
        pi * q.t
    };
}

template<int n1, int n2, int n3>
inline size_t hashXyt(const xyt& q) {
    static float a1 = n1 - 1 + 1.0 / n1,
                 a2 = n2 - 1 + 1.0 / n2,
                 a3 = n3 - 1 + 1.0 / n3;
    size_t i = round(q.x * a1),
           j = round(q.y * a2),
           k = round(q.t * a3);
    return i * (n2 * n3) + j * (n3) + k;
}

size_t HashXyt(const xyt& q) {
    return hashXyt<szx, szy, szt>(q);
}

static ReaderFunc<xyt, float, szxyt> FVxyt() {
    // call selected generator function
    return ReaderFunc<xyt, float, szxyt>(HashXyt);
}

template<int n1, int n2, int n3>
struct GradientGenerator {
    float A, B, C, D;

    float calScalar(const xyt& q) {
        static float 
            a1 = n1 - 1 + 1.0 / n1,
            a2 = n2 - 1 + 1.0 / n2,
            a3 = n3 - 1 + 1.0 / n3;
        float 
            dx = q.x - int(q.x * a1) / float(n1 - 1),
            dy = q.y - int(q.y * a2) / float(n2 - 1),
            dt = q.t - int(q.t * a3) / float(n3 - 1);
        return A * dx + B * dy + C * dt + D;
    }
};

template<int n1, int n2, int n3>
inline size_t _adapted_hashXyt(float x, float y, float t) {
    static float 
        a1 = n1 - 1 + 1.0 / n1,
        a2 = n2 - 1 + 1.0 / n2,
        a3 = n3 - 1 + 1.0 / n3;
    size_t 
        i = size_t(x * a1),
        j = size_t(y * a2),
        k = size_t(t * a3);
    return i * (n2 * n3) + j * n3 + k;
}

xyt interpolateGradient(float x, float y, float t) {
    static auto fv = FVxyt();
    /*
        fetch potential values of 4 points:
        (i,j,k), (i+1,j,k), (i,j+1,k), (i,j,k+1)
    */
    size_t ijk = _adapted_hashXyt<szx, szy, szt>(x, y, t);
    float 
        v000 = fv.data[ijk],
        v100 = fv.data[1 * szy * szt + ijk],
        v010 = fv.data[1 * szt + ijk],
        v001 = fv.data[1 + ijk];
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

template<int n1, int n2, int n3>
GradientGenerator<n1, n2, n3> interpolatePotential(const xyt& q) {
    static auto fv = FVxyt();
    /* 
        fetch potential values of 4 points: 
        (i,j,k), (i+1,j,k), (i,j+1,k), (i,j,k+1)
    */
    size_t ijk = hashXyt<n1, n2, n3>(q);
    float v000 = fv.data[ijk],
        v100 = fv.data[1 * n2 * n3 + ijk],
        v010 = fv.data[1 * n3 + ijk],
        v001 = fv.data[1 + ijk];
    /*
        solve the linear equation for (A,B,C,D):
        V(x,y,t) = A(x-x0) + B(y-y0) + C(t-t0) + D
    */
    float
        A = (-v000 + v100) * (n1 - 1),
        B = (-v000 + v010) * (n2 - 1),
        C = (-v000 + v100) * (n3 - 1),
        D = v000;
    /*
        the gradient: (A,B,C) is already obtained. (if only cauculate gradient, directly return)
        the value: A(x-x0) + B(y-y0) + C(t-t0) + D
    */
    return { A,B,C,D };
}




// interfaces

float _fpotential(float x, float y, float t) {
    xyt q = { x,y,t };
    return interpolatePotential<szx, szy, szt>(q).calScalar(q);
}

