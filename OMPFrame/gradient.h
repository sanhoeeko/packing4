#pragma once

#include"defs.h"
#include<vector>

using namespace std;

struct GradientBuffer
{
    VectorXf buffers[CORES];
    int N;
    int id;
    int sibling_id;
    
    GradientBuffer();
    GradientBuffer(int N);
    GradientBuffer(const GradientBuffer& obj);
    void clear();
    void joinTo(VectorXf* g);
};

struct EnergyBuffer
{
    float buffers[CORES];
    int N;
    int id;
    int sibling_id;

    EnergyBuffer();
    EnergyBuffer(int N);
    EnergyBuffer(const EnergyBuffer& obj);
    void clear();
    float sum();
};

struct PairInfo
{
    vector<ParticlePair> info_pp[CORES];
    vector<ParticlePair> info_pw[CORES];
    int N;
    int id;
    int sibling_id;

    PairInfo();
    PairInfo(int N);
    PairInfo(const PairInfo& obj);
    void clear();
    template<HowToCalGradient how> GradientBuffer* CalGradient();
    EnergyBuffer* CalEnergy();
};

#include "gradient_impl.h"