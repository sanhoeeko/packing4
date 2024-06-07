#pragma once

#include"defs.h"
#include"functional.h"
#include"scalar.h"

const int sz1d = 1ll << 20;

/*
    The range of (x,y,t): x = X/(2a) in [0,1), y = Y/(a+b) in [0,1), t = Theta/pi in [0,1)
    if szx == szy == szz, the maximal szx is 1024 for the sake of int.
*/
const int szx = 1ll << DIGIT_X;
const int szy = 1ll << DIGIT_Y;
const int szt = 1ll << DIGIT_T;
const int szxyt = szx * szy * szt;

float fsin(float x);
float fcos(float x);

float modpi(float x);

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
    D4ScalarFunc<szx, szy, szt>* fv;

    Rod(int n, float d);
    template<PotentialFunc what> void initPotential();

    // original definitions
    template<PotentialFunc what> float StandardPotential(const xyt& q);
    template<PotentialFunc what> XytPair StandardGradient(float x, float y, float t1, float t2);

    // auxiliary functions 
    xyt interpolateGradientSimplex(const xyt& q);
    float interpolatePotentialSimplex(const xyt& q);
    float potentialNoInterpolate(const xyt& q);

    // interfaces
    xyt gradient(const xyt& q);
    float potential(const xyt& q);
};

#include "potential_impl.h"