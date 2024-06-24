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
    vector<float> parallelEquilibrium(EquilibriumMethod func, int max_iterations);
};

extern Global* global;

void setGlobal();


template<typename ret_type> 
vector<ret_type> MapStates(ret_type(State::* func)(int), State** states, int max_iterations)
{
    vector<ret_type> res; res.resize(SIBLINGS);
#pragma omp parallel num_threads(SIBLINGS)
    {
        int idx = omp_get_thread_num();
        if (states[idx] != NULL) {
            res[idx] = (states[idx]->*func)(max_iterations);
        }
    }
    return res;
}