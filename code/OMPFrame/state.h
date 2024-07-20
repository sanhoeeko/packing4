#pragma once

#include"defs.h"
#include"functional.h"
#include"grid.h"
#include"gradient.h"

struct EllipseBoundary {
    float a, b;
    float a2, b2, inv_inner_a2, inv_inner_b2;
    bool if_a_less_than_b;

    EllipseBoundary(float a, float b);
    void setBoundary(float a, float b);
    bool maybeCollide(const xyt& particle);
    float distOutOfBoundary(const xyt& particle);
    void solveNearestPointOnEllipse(float x1, float y1, float& x0, float& y0);
    Maybe<ParticlePair> collide(int id, const xyt& particle);
};

struct State{
    VectorXf configuration;
    EllipseBoundary* boundary;
    vector<float> ge;               // ge can be either a) max gradient amplitudes, or b) energy records.

    int N;
    int sibling_id;

    Maybe<Grid*> grid;
    Maybe<PairInfo*> pair_info;

    // methods

    State(int N);
    State(int N, int sibling);
    State(int N, int sibling, VectorXf q, EllipseBoundary* b);
    void clearCache();
    void randomInitStateCC();
    float initAsDisks(int max_iterations);

    void setBoundary(float a, float b);
    void descent(float a, VectorXf& g);
    void loadFromData(float* data_src);
    void crashIfDataInvalid();
    float equilibriumGD(int max_iterations);
    float eqLineGD(int max_iterations);

    Grid* GridLocate();
    PairInfo* CollisionDetect();
    float CalEnergy();
    float meanDistance();
    float meanContactZ();

    template<HowToCalGradient how> VectorXf CalGradient() 
    {
        return this->CollisionDetect()->CalGradient<how>()->join();
    };
};