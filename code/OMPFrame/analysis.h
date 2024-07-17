#pragma once

#include "state.h"

vector<float> _landscapeAlongGradient(State* s, float max_stepsize, int samples);
vector<vector<float>> _landscapeOnGradientSections(State* s, float max_stepsize, int samples);