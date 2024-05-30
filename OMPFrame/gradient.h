#pragma once

#include"defs.h"
#include<vector>

using namespace std;

struct GradientAndEnergy
{
    VectorXf buffers[CORES];
    float energy_buffers[CORES];
    int N;
    int id;

    GradientAndEnergy(int N);
    void clear();
    void joinTo(VectorXf* g);
};

struct PairInfo
{
    vector<ParticlePair> info[CORES];
    int N;
    int id;

    PairInfo(int N);
    void clear();
    GradientAndEnergy* CalGradient();
    GradientAndEnergy* CalGradientAsDisks();
    GradientAndEnergy* CalEnergy();
};