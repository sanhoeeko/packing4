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

struct StateLoader {
    State* s_ref;
    State* s_temp;

    StateLoader(State* s);
    StateLoader* redefine(State* s);
    State* clear();
    State* setDescent(float a, VectorXf& g);
};

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
        Readonly: do not change the state in this function!
    */
    const float e = 1e-6f;  // to avoid the denominator being zero
    x[k + 1] = state->configuration;
    s[k] = x[k + 1] - x[k];
    g[k + 1] = state->CalGradient<Normal>();
    y[k] = g[k + 1] - g[k];
    rho[k] = 1.0f / (y[k].dot(s[k]) + e);
}

template<int m>
inline VectorXf L_bfgs<m>::CalDirection(State* state, int k)
{
    const float e = 1e-6f;  // to avoid the denominator being zero
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
        VectorXf z = (s[k - 1].dot(y[k - 1]) / (y[k - 1].dot(y[k - 1]) + e)) * q;
        for (int i = k - m; i <= k - 1; i++) {
            b[i] = rho[i] * y[i].dot(z);
            z += (a[i] - b[i]) * s[i];
        }
        return z;
    }
}

template<bool enable_line_search, bool enable_lbfgs>
inline float State::equilibrium(int max_iterations, float min_energy_slope)
{
    /*
        use `ge` as max gradient energies
    */
    float current_min_energy = CalEnergy();
    float step_size = enable_lbfgs ? 5e-3 : 1e-3;
    int turns_of_criterion = 0;

    const int m = 4;					// determine the precision of the inverse Hessian
    L_bfgs<m>* lbfgs = NULL;
    if constexpr (enable_lbfgs)lbfgs = new L_bfgs<m>(this);

    int energy_stride = (enable_line_search || enable_lbfgs) ? 10 : 1000;

    for (int i = 0; i < max_iterations; i++)
    {
        VectorXf g;
        // calculate the direction of descent
        if constexpr (enable_lbfgs) {
            g = lbfgs->CalDirection(this, i);
        }
        else {
            g = CalGradient<Normal>();
        }
        float gm = Modify(g);
        // gradient criterion
        if (gm < 1e-2) break;

        // calculate the step size
        if constexpr (enable_line_search) {
            try {
                step_size = BestStepSize(this, g, 0.1f);
            }
            catch (int exception) {  // exception == STEP_SIZE_TOO_SMALL
                step_size = 1e-2f;
            }
        }
        // descent
        descent(step_size, g);
        if constexpr (enable_lbfgs)lbfgs->update(this, i);

        // other criteria
        if ((i + 1) % energy_stride == 0) {
            float E = CalEnergy();
            ge.push_back(E);
            // energy criterion
            if (E < 1e-3) {
                break;
            }
            // step size (descent speed) criterion
            if (abs(1 - E / current_min_energy) < min_energy_slope) {
                if constexpr (enable_line_search) {
                    turns_of_criterion++;
                    if (turns_of_criterion >= 10)break;
                }
                else {
                    turns_of_criterion++;
                    if (turns_of_criterion >= 4) {
                        step_size *= 0.6;
                        turns_of_criterion = 0;
                        if (step_size < 1e-4)break;
                    }
                }
            }
            // moving average
            if (E < current_min_energy) {
                current_min_energy = (current_min_energy + E) / 2;
            }
        }
    }
    if constexpr (enable_lbfgs)delete lbfgs;
    return CalEnergy();
}