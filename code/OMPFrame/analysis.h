#pragma once

#include "state.h"
#include "optimizer.h"

vector<float> landscapeAlong(State* s, VectorXf& g, float max_stepsize, int samples);

template<HowToCalGradient how>
vector<float> _landscapeAlongGradient(State* s, float max_stepsize, int samples)
{
    VectorXf gradient = normalize(s->CalGradient<how>());
    return landscapeAlong(s, gradient, max_stepsize, samples);
}

vector<vector<float>> _landscapeOnGradientSections(State* s, float max_stepsize, int samples);