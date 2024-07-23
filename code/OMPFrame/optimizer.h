#pragma once

#include"defs.h"

#define STEP_SIZE_TOO_SMALL 0x0d000721

float maxGradientAbs(VectorXf& g);

/*
    returns the maximum amplitude (without normalization) of force
*/
float Modify(VectorXf& g);
VectorXf normalize(const VectorXf& g);

float ERoot(State* s, VectorXf& g, float expected_stepsize);
float BestStepSize(State* s, VectorXf& g, float max_stepsize);

template<int m>
struct L_bfgs {
    RollList<VectorXf, m> x, g, s, y;
    RollList<float, m> a, b, rho;

    L_bfgs(State* state) { init(state); }
    void init(State* s);
    void update(State* state, int k);
    VectorXf CalDirection(State* s, int k);
};

template<int m>
inline void L_bfgs<m>::init(State* state)
{
    const float a0 = 1e-4f;     // initial step size
    a[0] = a0;
    x[0] = state->configuration;
    g[0] = state->CalGradient<Normal>();
}

template<int m>
inline void L_bfgs<m>::update(State* state, int k)
{
    /*
        Readonly: do not change state in this function!
    */
    x[k + 1] = state->configuration;
    s[k] = x[k + 1] - x[k];
    g[k + 1] = state->CalGradient<Normal>();
    y[k] = g[k + 1] - g[k];
    rho[k] = 1.0f / y[k].dot(s[k]);
}

template<int m>
inline VectorXf L_bfgs<m>::CalDirection(State* state, int k)
{
    if (k < m) {
        return normalize(state->CalGradient<Normal>());
    }
    else {
        // solve for the descent direction: z
        VectorXf q = state->CalGradient<Normal>();
        for (int i = k - 1; i >= k - m; i--) {
            a[i] = rho[i] * s[i].dot(q);
            q -= a[i] * y[i];
        }
        VectorXf z = (s[k - 1].dot(y[k - 1]) / y[k - 1].dot(y[k - 1])) * q;
        for (int i = k - m; i <= k - 1; i++) {
            b[i] = rho[i] * y[i].dot(z);
            z += (a[i] - b[i]) * s[i];
        }
        return normalize(z);
    }
}
