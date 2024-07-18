#pragma once

#include"defs.h"


float maxGradientAbs(VectorXf& g);

/*
    returns the maximum amplitude (without normalization) of force
*/
float Modify(VectorXf& g);

std::pair<float, float> ERoot(State* s, VectorXf& g, float expected_stepsize);
float BestStepSize(State* s, VectorXf& g, float max_stepsize);