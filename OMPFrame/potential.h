#pragma once

#include"defs.h"
#include"functional.h"

const int sz1d = 1ll << 20;

/*
    The range of (x,y,t): x = X/(2a) in [0,1), y = Y/(a+b) in [0,1), t = Theta/pi in [0,1)
    if szx == szy == szz, the maximal szx is 1024 for the sake of size_t.
*/
const int szx = 1ll << DIGIT_X;
const int szy = 1ll << DIGIT_Y;
const int szt = 1ll << DIGIT_T;
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
    float
        a, b, c,
        a_padded, b_padded;
    ParticleShape() { ; }
    xyt transform(const xyt& q);
    xyt transform_signed(const xyt& q);
    xyt inverse(const xyt& q);
    bool isSegmentCrossing(const xyt& q);
};

struct Rod : ParticleShape {
    float
        rod_d,
        n_shift,
        inv_disk_R2;
    int n;
    ReaderFunc<xyt, float, szxyt>* fv;

    Rod(int n, float d);
    void initPotential();

    // original definitions
    float HertzianRodPotential(const xyt& q);
    XytPair HertzianGradientStandard(float x, float y, float t1, float t2);

    // auxiliary functions 
    xyt interpolateGradientSimplex(const xyt& q);
    float interpolatePotentialSimplex(const xyt& q);
    float potentialNoInterpolate(const xyt& q);

    // interfaces
    xyt gradient(const xyt& q);
    float potential(const xyt& q);
};
