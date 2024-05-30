#pragma once

#include "potential.h"
#include "state.h"

struct Global {
    Rod* rod;
    vector<State*> states;
};

void setGlobal();

void setRod(int n, float d);

void* createState(int N, float boundary_a, float boundary_b);

void* getStateData(void* state_ptr);

float fastPotential(float x, float y, float t);

float interpolatePotential(float x, float y, float t);

float precisePotential(float x, float y, float t);