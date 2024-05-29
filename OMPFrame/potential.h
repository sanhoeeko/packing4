#pragma once

#include"defs.h"
#include"functional.h"

const int sz1d = 1ll << 20;

const int szx = 1ll << 10;
const int szy = 1ll << 10;
const int szt = 1ll << 10;
const size_t szxyt = szx * szy * szt;

float fsin(float x);
float fcos(float x);
Matrix2f FU(float theta);

static inline float modpi(float x) {
    const float a = 1 / pi;
    float y = x * a;
    return y - std::floor(y);
}

/*
    Independent of the shape of anisotropic particles
*/

struct ParticleShape {
    float a, b, c;
    ParticleShape() { ; }
    ParticleShape(float a, float b, float c) : a(a), b(b), c(c) { ; }
    xyt transform(const xyt& q) {
        return
        {
            abs(q.x) / (2 * a),
            abs(q.y) / (a + b),
            (q.x > 0) ^ (q.y > 0) ? modpi(q.t) : 1 - modpi(q.t)     // doubted
        };
    }
    xyt inverse(const xyt& q) {
        return
        {
            (2 * a) * q.x,
            (a + b) * q.y,
            pi * q.t
        };
    }
    bool isSegmentCrossing(const xyt& q) {
        return 
            q.y < c * fsin(q.t) && 
            q.y * fcos(q.t) > (q.x - c) * fsin(q.t);
    }
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
    float potential(const xyt& q);
};

size_t HashXyt(const xyt& q);


template<int n1, int n2, int n3>
struct GradientGenerator {
    float A, B, C, D;

    float calScalar(const xyt& q) {
        const float
            a1 = n1 - 1,
            a2 = n2 - 1,
            a3 = n3 - 1;
        float
            dx = q.x - floor(q.x * a1) / a1,
            dy = q.y - floor(q.y * a2) / a2,
            dt = q.t - floor(q.t * a3) / a3;
        return A * dx + B * dy + C * dt + D;
    }
};