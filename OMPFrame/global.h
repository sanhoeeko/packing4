#pragma once

#include "defs.h"
#include "potential.h"
#include "state.h"

struct Global {
    Rod* rod;
    PotentialFunc pf;
};

extern Global* global;

void setGlobal();