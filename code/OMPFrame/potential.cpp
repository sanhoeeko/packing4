#include "pch.h"

#include"potential.h"
#include"functional.h"
#include"array.h"

#include<math.h>

float modpi(float x)
{
    const float a = 1 / pi;
    float y = x * a;
    return y - std::floor(y);
}

template<int capacity>
static inline int hashFloat2Pi(const float& x) {
    /*
        using "bitwise and" for fast modulo. 
        require: `capacity` is a power of 2.
    */
    const float a = capacity / (2 * pi);
    const int mask = capacity - 1;
    return (int)(a * x) & mask;
}

template<>
int anyHasher<float, _h2pi>(const float& x) {
    return hashFloat2Pi<sz1d>(x);
}

static inline float _sin(const float& x) { return sin(x); }
static inline float _cos(const float& x) { return cos(x); }

static LookupFunc<float, float, sz1d, _h2pi> FSin() {
    static vector<float> xs = linspace(0, 2 * pi, sz1d);
    static auto f = new LookupFunc<float, float, sz1d, _h2pi>(_sin, xs);
    return *f;
}

static LookupFunc<float, float, sz1d, _h2pi> FCos() {
    static vector<float> xs = linspace(0, 2 * pi, sz1d);
    static auto f = new LookupFunc<float, float, sz1d, _h2pi>(_cos, xs);
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
        hi = i <= X ? 1 : -1,   // do not use '<', because X and i can be both 0.0f and hi = -1 causes an illegal access
        hj = j <= Y ? 1 : -1,
        hk = k <= T ? 1 : -1;
    float
        v000 = fv->data[i][j][k],
        v100 = fv->data[i + hi][j][k],
        v010 = fv->data[i][j + hj][k],
        v001 = fv->data[i][j][k + hk];
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
        hi = dx >= 0 ? 1 : -1,
        hj = dy >= 0 ? 1 : -1,
        hk = dt >= 0 ? 1 : -1;
    float
        v000 = fv->data[i][j][k],
        v100 = fv->data[i + hi][j][k],
        v010 = fv->data[i][j + hj][k],
        v001 = fv->data[i][j][k + hk];
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
