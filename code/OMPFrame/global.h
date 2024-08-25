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

    State* newState(int N);
};

extern Global* global;

void setGlobal();
