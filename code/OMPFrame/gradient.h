#pragma once

#include"defs.h"
#include"functional.h"
#include"graph.h"
#include<vector>

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
    bool is_filtered;

    Maybe<GradientBuffer*> g_buffer;
    Maybe<EnergyBuffer*> e_buffer;
    Maybe<Graph<neighbors>*> graph;

    PairInfo();
    PairInfo(int N);
    void clear();
    template<HowToCalGradient how> GradientBuffer* CalGradient();
    EnergyBuffer* CalEnergy();
    PairInfo* filterAsRods(float gamma);
    int contactNumberZ(float gamma);
    float meanDistance(float gamma);
    Graph<neighbors>* toGraph(float gamma);
};

#include "gradient_impl.h"