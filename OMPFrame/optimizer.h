#pragma once

#include"defs.h"


float maxGradientAbs(VectorXf& g);

/*
    returns the maximum amplitude (without normalization) of force
*/
float Modify(VectorXf& g);