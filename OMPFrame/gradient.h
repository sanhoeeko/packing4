#pragma once

#include"defs.h"
#include<vector>

using namespace std;

struct GradientAndEnergy
{
    VectorXf buffers[CORES];
    VectorXf gradient;
    int N;
    float energy;

    void clear();
    void joinTo(VectorXf* g);
};

struct PairInfo
{
    vector<ParticlePair> info[CORES];

    void clear();
    GradientAndEnergy* CalGradient();
};