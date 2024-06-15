#pragma once

#include<omp.h>

#undef min
#undef max

#ifdef _WIN32
#include<Eigen/Dense>
#else
#include"Eigen/Dense"
#endif
using namespace Eigen;

#define CORES 4
#define NAN_IF_PENETRATE false
#define ENABLE_NAN_CHECK true

#define DIGIT_X 9
#define DIGIT_Y 9
#define DIGIT_T 9
#define ENERGY_STRIDE 1000

const float pi = 3.141592654;

enum HowToCalGradient{ Normal, AsDisks, Test, HowToCalGradient_Count };
enum PotentialFunc{ Hertzian, ScreenedCoulomb, PotentialFunc_Count };
enum HashFunc { _h2pi, _h4 };

struct xyt { 
    float x, y, t; 
    void operator+=(const xyt& o);
    void operator-=(const xyt& o);
    float amp2();
};

struct XytPair {
    xyt first, second;
};

struct ParticlePair { int id1, id2; float x, y, t1, t2; };
