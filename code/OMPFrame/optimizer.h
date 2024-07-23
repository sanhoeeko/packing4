#pragma once

#include"defs.h"

#define STEP_SIZE_TOO_SMALL 0x0d000721

float maxGradientAbs(VectorXf& g);

/*
    returns the maximum amplitude (without normalization) of force
*/
float Modify(VectorXf& g);
VectorXf normalize(const VectorXf& g);

float ERoot(State* s, VectorXf& g, float expected_stepsize);
float BestStepSize(State* s, VectorXf& g, float max_stepsize);