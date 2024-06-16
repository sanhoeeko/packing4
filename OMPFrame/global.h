#pragma once

#include "defs.h"
#include "potential.h"
#include "state.h"

struct Global {
    Rod* rod;
    PotentialFunc pf;

    int newSibling();

private:
    int sibling_num;
};

extern Global* global;

void setGlobal();