#pragma once

#include"defs.h"
#include"functional.h"
#include<vector>

using namespace std;

struct GradientBuffer
{
    VectorXf buffers[CORES];
    int N;

    Maybe<VectorXf*> result;
    
    GradientBuffer();
    GradientBuffer(int N);
    void clear();
    VectorXf join();
};

struct EnergyBuffer
{
    float buffers[CORES];
    int N;

    Maybe<float> result;

    EnergyBuffer();
    EnergyBuffer(int N);
    void clear();
    float sum();
};

struct PairInfo
{
    vector<ParticlePair> info_pp[CORES];
    vector<ParticlePair> info_pw[CORES];

    int N;

    Maybe<GradientBuffer*> g_buffer;
    Maybe<EnergyBuffer*> e_buffer;

    PairInfo();
    PairInfo(int N);
    void clear();
    template<HowToCalGradient how> GradientBuffer* CalGradient();
    EnergyBuffer* CalEnergy();
    int contactNumberZ();
    float meanDistance();
};

#include "gradient_impl.h"