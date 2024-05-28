#pragma once

#include"defs.h"
#include"functional.h"
#include"grid.h"


struct EllipseBoundary {
    float a, b;
    bool maybeCollide(const xyt& particle);
    Maybe<ParticlePair> collide(const xyt& particle);
};

struct State{
    VectorXf configuration;
    EllipseBoundary* boundary;
    int N;

    State(int N);
    State(VectorXf q, EllipseBoundary* b, int N);
    State GradientDescent(float a);

    Grid* GridLocate();
    PairInfo* CollisionDetect();
    VectorXf* CalGradient();
};

State randomInitCC(int N, EllipseBoundary* b);