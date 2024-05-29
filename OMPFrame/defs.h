#pragma once

#include<omp.h>

#ifdef WIN32
#include<Eigen/Dense>
#else
#include"Eigen/Dense"
#endif
using namespace Eigen;

#define CORES 4
#define NAN_IF_PENETRATE false

const float pi = 3.141592654;

struct xyt { 
    float x, y, t; 
    void operator+=(const xyt& o);
    void operator-=(const xyt& o);
};

struct ParticlePair { int id1, id2; float x, y, t1, t2; };