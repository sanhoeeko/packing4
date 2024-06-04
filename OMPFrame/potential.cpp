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

Matrix2f FU(float theta) { 
    static auto _u = FRotation();
    return _u(theta); 
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

xyt Rod::interpolateGradientSimplex(const xyt& q) {
    /*
        fetch potential values of 4 points:
        (i,j,k), (i+1,j,k), (i,j+1,k), (i,j,k+1)
    */
    if (q.y >= 1)return { 0,0,0 };
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
        C = (-v000 + v001) * (szt - 1);
        // D = v000;
    /*
        the gradient: (A,B,C) is already obtained. (if only cauculate gradient, directly return)
        the value: A(x-x0) + B(y-y0) + C(t-t0) + D
    */
    return { A,B,C };
}

float Rod::interpolatePotentialSimplex(const xyt& q) {
    const float
        a1 = szx - 1,
        a2 = szy - 1,
        a3 = szt - 1;
    /*
        fetch potential values of 4 points:
        (i,j,k), (i+1,j,k), (i,j+1,k), (i,j,k+1)
    */
    if (q.y >= 1)return 0;
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
        A = (-v000 + v100) * a1,
        B = (-v000 + v010) * a2,
        C = (-v000 + v001) * a3,
        D = v000;
    /*
        the energy: A(x-x0) + B(y-y0) + C(t-t0) + D
        since x0 <- floor'(x), there must be x > x0, y > y0, t > t0
    */
    float
        dx = q.x - floor(q.x * a1) / a1,
        dy = q.y - floor(q.y * a2) / a2,
        dt = q.t - floor(q.t * a3) / a3;
    return A * dx + B * dy + C * dt + D;
}
