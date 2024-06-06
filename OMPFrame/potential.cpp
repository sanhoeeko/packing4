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

static inline float modpi(float x)
{
    const float a = 1 / pi;
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

size_t HashXyt(const xyt& q) {
    return hashXyt<szx, szy, szt>(q);
}

xyt ParticleShape::transform(const xyt& q)
{
    return
    {
        abs(q.x) / (2 * a_padded),
        abs(q.y) / (a_padded + b_padded),
        (q.x > 0) ^ (q.y > 0) ? modpi(q.t) : 1 - modpi(q.t)
    };
}

xyt ParticleShape::transform_signed(const xyt& g)
{
    return
    {
        g.x / (2 * a_padded),
        g.y / (a_padded + b_padded),
        g.t / pi
    };
}

xyt ParticleShape::inverse(const xyt& q)
{
    return
    {
        (2 * a_padded) * q.x,
        (a_padded + b_padded) * q.y,
        pi * q.t
    };
}

bool ParticleShape::isSegmentCrossing(const xyt& q)
{
    return
        q.y < c * fsin(q.t) &&
        q.y * fcos(q.t) >(q.x - c) * fsin(q.t);
}

xyt Rod::interpolateGradientSimplex(const xyt& q) 
{
    if (q.y >= 1)return { 0,0,0 };
    const float 
        a1 = szx - 1,
        a2 = szy - 1,
        a3 = szt - 1;
    /*
        fetch potential values of 4 points:
        (i,j,k), (i ¡À 1,j,k), (i,j ¡À 1,k), (i,j,k ¡À 1)
    */
    float
        X = q.x * a1,
        Y = q.y * a2,
        T = q.t * a3;
    int 
        i = round(X),
        j = round(Y),
        k = round(T);
    int
        hi = i < X ? 1 : -1,
        hj = j < Y ? 1 : -1,
        hk = k < T ? 1 : -1;
    size_t
        ijk = (size_t)i * (szy * szt) + j * szt + k;
    float 
        v000 = fv->data[ijk],
        v100 = fv->data[(size_t)hi * szy * szt + ijk],
        v010 = fv->data[(size_t)hj * szt + ijk],
        v001 = fv->data[(size_t)hk + ijk];
    /*
        solve the linear equation for (A,B,C,D):
        V(x,y,t) = A(x-x0) + B(y-y0) + C(t-t0) + D
    */
    float
        A = (-v000 + v100) * a1 * hi,
        B = (-v000 + v010) * a2 * hj,
        C = (-v000 + v001) * a3 * hk;
        // D = v000;
    /*
        the gradient: (A,B,C) is already obtained. (if only cauculate gradient, directly return)
        the value: A(x-x0) + B(y-y0) + C(t-t0) + D
    */
    return { A,B,C };
}

float Rod::interpolatePotentialSimplex(const xyt& q) 
{
    if (q.y >= 1)return 0;
    const float
        a1 = szx - 1,
        a2 = szy - 1,
        a3 = szt - 1;
    /*
        fetch potential values of 4 points:
        (i,j,k), (i ¡À 1,j,k), (i,j ¡À 1,k), (i,j,k ¡À 1)
    */
    float
        X = q.x * a1,
        Y = q.y * a2,
        T = q.t * a3;
    int
        i = round(X),
        j = round(Y),
        k = round(T);
    float
        dx = (X - i) / a1,
        dy = (Y - j) / a2,
        dt = (T - k) / a3;
    int
        hi = dx > 0 ? 1 : -1,
        hj = dy > 0 ? 1 : -1,
        hk = dt > 0 ? 1 : -1;
    size_t
        ijk = (size_t)i * (szy * szt) + j * szt + k;
    float
        v000 = fv->data[ijk],
        v100 = fv->data[(size_t)hi * szy * szt + ijk],
        v010 = fv->data[(size_t)hj * szt + ijk],
        v001 = fv->data[(size_t)hk + ijk];
    /*
        solve the linear equation for (A,B,C,D):
        V(x,y,t) = A(x-x0) + B(y-y0) + C(t-t0) + D
    */
    float
        A = (-v000 + v100) * a1 * hi,
        B = (-v000 + v010) * a2 * hj,
        C = (-v000 + v001) * a3 * hk,
        D = v000;
    /*
        the energy: A(x-x0) + B(y-y0) + C(t-t0) + D
        since x0 <- floor'(x), there must be x > x0, y > y0, t > t0
    */
    return A * dx + B * dy + C * dt + D;
}
