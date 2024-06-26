#pragma once

#include "defs.h"
#include "potential.h"
#include "state.h"

typedef float (State::* EquilibriumMethod)(int);


struct Global {
    vector<State*> states;
    Rod* rod;
    PotentialFunc pf;

    State* newState(int N);
};

extern Global* global;

void setGlobal();
