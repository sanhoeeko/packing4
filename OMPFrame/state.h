#pragma once

#include"defs.h"
#include"functional.h"
#include"grid.h"

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
    VectorXf gradient;
    EllipseBoundary* boundary;
    vector<float> ge;               // ge can either be: a) max gradient amplitudes; b) energy records.
    int N;
    int id;
    int sibling_id;

    State(int N, int sibling);
    State(VectorXf q, EllipseBoundary* b, int N, int sibling);
    void randomInitStateCC();
    float initAsDisks(int max_iterations);

    void setBoundary(float a, float b);
    void descent(float a, VectorXf& g);
    void crashIfDataInvalid();
    float equilibriumGD(int max_iterations);

    Grid* GridLocate();
    PairInfo* CollisionDetect();
    template<HowToCalGradient how> VectorXf CalGradient();
    float CalEnergy();
};