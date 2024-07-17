#include "pch.h"
#include "analysis.h"
#include <random>

VectorXf randomUnitVector(int n) {
    /*
        using Gaussian distribution: only for angular uniformity.
    */
    std::default_random_engine generator;
    std::normal_distribution<float> distribution(0.0, 1.0);
    Eigen::VectorXf v(n);
    for (int i = 0; i < n; i++) {
        v[i] = distribution(generator);
    }
    return v / v.norm();
}

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

/*
    make a 2D energy landscape on a random section,
    one direction of which is the gradient.
    return: float[2 n + 1][n]
*/
vector<vector<float>> _landscapeOnGradientSections(State* s, float max_stepsize, int samples)
{
    vector<vector<float>> res; res.reserve(2 * samples + 1);
    float d_stepsize = max_stepsize / samples;
    VectorXf g = s->CalGradient<Normal>();
    VectorXf u = randomUnitVector(s->N);
    State* s_temp = new State(s->N);
    s_temp->boundary = s->boundary;

    for (int i = -samples; i <= samples; i++) {
        s_temp->loadFromData(s->configuration.data());
        s_temp->descent(i * d_stepsize, u);
        res.push_back(landscapeAlong(s_temp, g, max_stepsize, samples));
    }
    return res;
}