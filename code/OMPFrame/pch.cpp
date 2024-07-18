// pch.cpp: 与预编译标头对应的源文件

#include "pch.h"

// 当使用预编译的头时，需要使用此源文件，编译才能成功。

#include"global.h"

void init() {
    omp_set_num_threads(CORES);
    omp_set_nested(1);  // Enable nested parallelism
    setGlobal();
}

State* Global::newState(int N)
{
    State* state = new State(N, global->states.size());
    global->states.push_back(state);
    return state;
}