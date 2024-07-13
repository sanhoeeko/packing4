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
    vector<float> ge;               // ge can either be: a) max gradient amplitudes; b) energy records.

    int N;
    int sibling_id;

    Maybe<Grid*> grid;
    Maybe<PairInfo*> pair_info;

    // methods

    void commonInit(int N);
    State(int N, int sibling);
    State(VectorXf q, EllipseBoundary* b, int N, int sibling);
    void clearCache();
    void randomInitStateCC();
    float initAsDisks(int max_iterations);

    void setBoundary(float a, float b);
    void descent(float a, VectorXf& g);
    void loadFromData(float* data_src);
    void crashIfDataInvalid();
    float equilibriumGD(int max_iterations);

    Grid* GridLocate();
    PairInfo* CollisionDetect();
    float CalEnergy();

    template<HowToCalGradient how> VectorXf CalGradient() 
    {
        return this->CollisionDetect()->CalGradient<how>()->join();
    };
};