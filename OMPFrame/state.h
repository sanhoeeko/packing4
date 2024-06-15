#pragma once

#include"defs.h"
#include"functional.h"
#include"grid.h"

struct EllipseBoundary {
    float a, b;
    float a2, b2, inv_inner_a2, inv_inner_b2;
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
    vector<float> max_gradient_amps;
    int N;
    int id;

    State(int N);
    State(VectorXf q, EllipseBoundary* b, int N);
    void randomInitStateCC();
    void initAsDisks();

    void setBoundary(float a, float b);
    void descent(float a, VectorXf* g);
    float equilibriumGD(int max_iterations);

    Grid* GridLocate();
    PairInfo* CollisionDetect();
    template<HowToCalGradient how> VectorXf* CalGradient();
    float CalEnergy();
};