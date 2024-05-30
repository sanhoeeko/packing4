#pragma once

#include"defs.h"
#include"functional.h"
#include"grid.h"


struct EllipseBoundary {
    float a, b;
    EllipseBoundary(float a, float b);
    bool maybeCollide(const xyt& particle);
    Maybe<ParticlePair> collide(const xyt& particle);
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

    void descent(float a, VectorXf* g);
    State GradientDescent(float a);

    Grid* GridLocate();
    PairInfo* CollisionDetect();
    VectorXf* CalGradient();
};
