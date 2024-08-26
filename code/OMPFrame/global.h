#pragma once

#include "defs.h"
#include "potential.h"
#include "state.h"
#include "objpool.h"

typedef float (State::* EquilibriumMethod)(int);

const int simulators = 2 * CORES;

struct Global {
    ObjectPool<State, simulators> states;
    Rod* rod;
    PotentialFunc pf;
    float power_of_potential;

    State* newState(int N);
    void checkPowerOfPotential();
};

extern Global* global;

void setGlobal();
