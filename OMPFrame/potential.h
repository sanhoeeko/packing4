#pragma once

#include"defs.h"
#include"functional.h"

const int sz1d = 1ll << 20;

const int szx = 1ll << 11;
const int szy = 1ll << 10;
const int szt = 1ll << 9;
const size_t szxyt = szx * szy * szt;

float fsin(float x);
float fcos(float x);
Matrix2f FU(float theta);

/*
    Independent of the shape of anisotropic particles
*/
struct Gate {
    float a, b;
    Gate();
    Gate(float a, float b);
    xyt transform(const xyt& q);
    xyt inverse(const xyt& q);
};

struct Rod {
    ReaderFunc<xyt, float, szxyt>* fv;
    Gate gate;
    float rod_d, a, b;
    float shift;
    int n;

    Rod(int n, float d);
    float HertzianRodPotential(const xyt& q);
    float hertzian_rod_01(const xyt& q);
    void initPotential();
    float potential(const xyt& q);
};

size_t HashXyt(const xyt& q);

xyt interpolateGradient(float x, float y, float t);