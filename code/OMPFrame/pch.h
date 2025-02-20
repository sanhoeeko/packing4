﻿// pch.h: 这是预编译标头文件。
// 下方列出的文件仅编译一次，提高了将来生成的生成性能。
// 这还将影响 IntelliSense 性能，包括代码完成和许多代码浏览功能。
// 但是，如果此处列出的文件中的任何一个在生成之间有更新，它们全部都将被重新编译。
// 请勿在此处添加要频繁更新的文件，这将使得性能优势无效。

#ifndef PCH_H
#define PCH_H

// 添加要在此处预编译的标头
#include "framework.h"

#include "defs.h"
#include "global.h"
#ifdef _WIN32
    #define DLLEXPORT extern "C" __declspec(dllexport)
#else
    #define DLLEXPORT extern "C"
#endif

// simulation
DLLEXPORT void init();
DLLEXPORT void setEnums(int potential_func);
DLLEXPORT void setPotentialPower(float power);
DLLEXPORT void declareRod(int n, float d);
DLLEXPORT void setRod(int n, float d, int threads);
DLLEXPORT void* createState(int N, float boundary_a, float boundary_b);
DLLEXPORT void initStateAsDisks(void* state_ptr);
DLLEXPORT void setBoundary(void* state_ptr, float boundary_a, float boundary_b);
DLLEXPORT void singleStep(void* state_ptr, int mode, float step_size);
DLLEXPORT float equilibriumGD(void* state_ptr, int max_iterations);
DLLEXPORT float eqLineGD(void* state_ptr, int max_iterations);
DLLEXPORT float eqLBFGS(void* state_ptr, int max_iterations);
DLLEXPORT float eqMix(void* state_ptr, int max_iterations);

// fetching data
DLLEXPORT void* getStateData(void* state_ptr);
DLLEXPORT int getStateIterations(void* state_ptr);
DLLEXPORT void* getStateMaxGradOrEnergy(void* state_ptr);
DLLEXPORT void* getStateResidualForce(void* state_ptr);
DLLEXPORT float getStateMaxResidualForce(void* state_ptr);
DLLEXPORT int getSiblingId(void* state_ptr);

// load data
DLLEXPORT void setStateData(void* state_ptr, void* data_src);
DLLEXPORT void* loadState(void* data_src, int N, float boundary_a, float boundary_b);
DLLEXPORT void freeState(void* state_ptr);
DLLEXPORT void renewState(void* state_ptr);

// built-in IO
DLLEXPORT void readPotential(int n, float d);
DLLEXPORT void writePotential();
DLLEXPORT int getPotentialId();

// analysis
DLLEXPORT float* landscapeAlongGradient(void* state_ptr, float max_stepsize, int samples);
DLLEXPORT float* landscapeLBFGS(void* state_ptr, float max_stepsize, int samples);
DLLEXPORT float* landscapeOnGradientSections(void* state_ptr, float max_stepsize, int samples);
DLLEXPORT float meanDistance(void* state_ptr, float gamma);
DLLEXPORT float meanContactZ(void* state_ptr, float gamma);
DLLEXPORT float meanS(void* state_ptr, float gamma);
DLLEXPORT float absPhi(void* state_ptr, float gamma, int p);
DLLEXPORT float* Si(void* state_ptr, float gamma);
DLLEXPORT float* neighborAngleDist(void* state_ptr, float gamma, int bins);

// test of algorithms
DLLEXPORT float fastPotential(float x, float y, float t);
DLLEXPORT float interpolatePotential(float x, float y, float t);
DLLEXPORT float precisePotential(float x, float y, float t);
DLLEXPORT float* interpolateGradient(float x, float y, float t);
DLLEXPORT float* gradientReference(float x, float y, float t1, float t2);
DLLEXPORT float* gradientTest(float x, float y, float t1, float t2);
DLLEXPORT float* getMirrorOf(float A, float B, float x, float y, float t);
DLLEXPORT float testERoot(void* state_ptr, float max_stepsize);
DLLEXPORT float testBestStepSize(void* state_ptr, float max_stepsize);

// test in cpp
void runTest();

#endif //PCH_H
