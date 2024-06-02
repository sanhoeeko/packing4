#include "pch.h"
#include "global.h"

Global* global;

void setGlobal()
{
    global = new Global();
}

void setRod(int n, float d)
{
    global->rod = new Rod(n, d);
    global->rod->initPotential();
}

void* createState(int N, float boundary_a, float boundary_b)
{ 
    auto state = new State(N);
    state->boundary = new EllipseBoundary(boundary_a, boundary_b);
    state->randomInitStateCC();
    global->states.push_back(state);
    return state;
}

void* getStateData(void* state_ptr)
{
    State* s = reinterpret_cast<State*>(state_ptr);
    return s->configuration.data();
}

int getStateIterations(void* state_ptr)
{
    State* s = reinterpret_cast<State*>(state_ptr);
    return s->max_gradient_amps.size();
}

DLLEXPORT void* getStateResidualForce(void* state_ptr)
{
    State* s = reinterpret_cast<State*>(state_ptr);
    return s->CalGradient<AsDisks>()->data();
}

void* getStateMaxGradients(void* state_ptr)
{
    State* s = reinterpret_cast<State*>(state_ptr);
    return s->max_gradient_amps.data();
}

void initStateAsDisks(void* state_ptr) {
    State* s = reinterpret_cast<State*>(state_ptr);
    return s->initAsDisks();
}

void setBoundary(void* state_ptr, float boundary_a, float boundary_b)
{
    State* s = reinterpret_cast<State*>(state_ptr);
    s->boundary->setBoundary(boundary_a, boundary_b);
}

void equilibriumGD(void* state_ptr)
{
    State* s = reinterpret_cast<State*>(state_ptr);
    s->equilibriumGD();
}

float fastPotential(float x, float y, float t)
{
    return global->rod->potentialNoInterpolate({ x,y,t });
}

float interpolatePotential(float x, float y, float t)
{
    if (abs(x) > 2 * global->rod->a || abs(y) > global->rod->a + global->rod->b) {
        return 0;
    }
    else {
        return global->rod->potential({ x,y,t });
    }
}

float precisePotential(float x, float y, float t)
{
    return global->rod->HertzianRodPotential({
        abs(x),
        abs(y),
        (x > 0) ^ (y > 0) ? t : pi - t,
    });
}

float* interpolateGradient(float x, float y, float t)
{
    static float arr[3];
    if (abs(x) > 2 * global->rod->a || abs(y) > global->rod->a + global->rod->b) {
        arr[0] = 0; arr[1] = 0; arr[2] = 0;
    }
    else {
        xyt g = global->rod->gradient({ x,y,t });
        arr[0] = g.x; arr[1] = g.y; arr[2] = g.t;
    }
    return arr;
}
