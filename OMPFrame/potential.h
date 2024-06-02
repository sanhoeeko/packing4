#pragma once

#include"defs.h"
#include"functional.h"

const int sz1d = 1ll << 20;

/*
    The range of (x,y,t): x = X/(2a) in [0,1), y = Y/(a+b) in [0,1), t = Theta/pi in [0,1)
    if szx == szy == szz, the maximal szx is 1024 for the sake of size_t.
*/
const int szx = 1ll << 8;
const int szy = 1ll << 8;
const int szt = 1ll << 8;
const size_t szxyt = szx * szy * szt;

float fsin(float x);
float fcos(float x);
Matrix2f FU(float theta);
float d_isotropicSq_r(float x2);

float modpi(float x);
size_t HashXyt(const xyt& q);

template<int n1, int n2, int n3>    // cannot be moved to impl header
size_t hashXyt(const xyt& q) {
    const float a1 = n1 - 1,
        a2 = n2 - 1,
        a3 = n3 - 1;
    size_t i = round(q.x * a1),
        j = round(q.y * a2),
        k = round(q.t * a3);
    return i * (n2 * n3) + j * (n3)+k;
}

template<int n1, int n2, int n3>    // cannot be moved to impl header
size_t hashXytFloor(const xyt& q) {
    const float a1 = n1 - 1,
        a2 = n2 - 1,
        a3 = n3 - 1;
    size_t i = size_t(q.x * a1),    // floor
        j = size_t(q.y * a2),
        k = size_t(q.t * a3);
    return i * (n2 * n3) + j * (n3)+k;
}

/*
    Base class. It is ndependent of the shape of anisotropic particles
*/
struct ParticleShape {
    float a, b, c;
    ParticleShape() { ; }
    ParticleShape(float a, float b, float c) : a(a), b(b), c(c) { ; }
    xyt transform(const xyt& q);
    xyt inverse(const xyt& q);
    bool isSegmentCrossing(const xyt& q);
};

struct Rod : ParticleShape {
    ReaderFunc<xyt, float, szxyt>* fv;
    float rod_d;
    float n_shift;
    int n;

    Rod(int n, float d);
    float HertzianRodPotential(const xyt& q);
    xyt interpolateGradientSimplex(const xyt& q);
    float interpolatePotentialSimplex(const xyt& q);
    void initPotential();
    float potentialNoInterpolate(const xyt& q);
    xyt gradient(const xyt& q);
    float potential(const xyt& q);
};
