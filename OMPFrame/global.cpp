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

void* getStateMaxGradient(void* state_ptr)
{
    State* s = reinterpret_cast<State*>(state_ptr);
    return s->max_gradient_amps.data();
}

void initStateAsDisks(void* state_ptr) {
    State* s = reinterpret_cast<State*>(state_ptr);
    return s->initAsDisks();
}


float fastPotential(float x, float y, float t)
{
    return global->rod->potentialNoInterpolate({ x,y,t });
}

float interpolatePotential(float x, float y, float t)
{
    return global->rod->potential({ x,y,t });
}

float precisePotential(float x, float y, float t)
{
    return global->rod->HertzianRodPotential({
        abs(x),
        abs(y),
        (x > 0) ^ (y > 0) ? t : pi - t,
    });
}
