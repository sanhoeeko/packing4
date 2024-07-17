#include "pch.h"
#include "analysis.h"

vector<float> landscapeAlong(State* s, VectorXf& g, float max_stepsize, int samples)
{
    vector<float> res; res.reserve(samples);
    float d_stepsize = max_stepsize / samples;
    State* s_temp = new State(s->N);
    s_temp->boundary = s->boundary;

    for (int i = 0; i < samples; i++) {
        s_temp->loadFromData(s->configuration.data());
        s_temp->descent((i + 1) * d_stepsize, g);
        float energy = s_temp->CalEnergy();
        res.push_back(energy);
    }
    return res;
}

vector<float> _landscapeAlongGradient(State* s, float max_stepsize, int samples) 
{
    VectorXf gradient = s->CalGradient<Normal>();
    return landscapeAlong(s, gradient, max_stepsize, samples);
}