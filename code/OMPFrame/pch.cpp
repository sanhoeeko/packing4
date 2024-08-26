// pch.cpp: 与预编译标头对应的源文件

#include "pch.h"

// 当使用预编译的头时，需要使用此源文件，编译才能成功。

#include "global.h"

void init() {
    omp_set_num_threads(CORES);
    omp_set_nested(1);                          // Enable nested parallelism
    setGlobal();
}

State* Global::newState(int N)
{
    int id = states.getAvailableId();
    if (id == -1) {
        cout << "New state allocation fails!" << endl;
        throw -1;
    }
    if (states[id] == NULL) {
        states[id] = new State(N);
    }
    else if (states[id]->N != N){
        delete states[id];
        states[id] = new State(N);
    }
    return states[id];
}

void Global::checkPowerOfPotential()
{
    if (this->pf == GeneralizedHertzian && this->power_of_potential == 0) {
        throw "Power of potential not set!";
    }
}

void runTest()
{
    float q = 1 - 1e-3;
    init();
    setEnums(PotentialFunc::ScreenedCoulomb);
    setRod(6, 0.05, 4);
    State* state = (State*)createState(200, 20, 20);
    initStateAsDisks(state);
    for (int i = 0; i < 1000; i++) {
        setBoundary(state, state->boundary->a, state->boundary->b * q);
        float energy = eqMix(state, (int)4e5);
        cout << "step: " << i << ", energy: " << energy << endl;
    }
}