#pragma once

#include "potential.h"
#include "state.h"

struct Global {
    Rod* rod;
    vector<State*> states;
};

extern Global* global;

void setGlobal();