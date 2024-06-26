#pragma once

#include "defs.h"
#include "potential.h"
#include "state.h"

typedef float (State::* EquilibriumMethod)(int);


struct Global {
    State* states[SIBLINGS];
    Rod* rod;
    PotentialFunc pf;
    int n_states;

    int newSibling();
    State* newState(int N);
};

extern Global* global;

void setGlobal();
