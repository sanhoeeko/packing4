#pragma once

#include"defs.h"
#include<vector>

using namespace std;

struct GradientBuffer
{
    VectorXf buffers[CORES];
    int N;
    int id;
    

    GradientBuffer(int N);
    void clear();
    void joinTo(VectorXf* g);
};

struct EnergyBuffer
{
    float buffers[CORES];
    int N;
    int id;

    EnergyBuffer(int N);
    void clear();
    float sum();
};

struct PairInfo
{
    vector<ParticlePair> info_pp[CORES];
    vector<ParticlePair> info_pw[CORES];
    int N;
    int id;

    PairInfo(int N);
    void clear();
    template<HowToCalGradient how> GradientBuffer* CalGradient();
    EnergyBuffer* CalEnergy();
};

#include "gradient_impl.h"