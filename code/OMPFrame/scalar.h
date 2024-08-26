#pragma once

#include "defs.h"

const int sz1d = 1ll << 20;

template<PotentialFunc what> float scalarPotential(float x2);

template<PotentialFunc what> float potentialDR(float x2);

extern LookupFunc<float, float, sz1d, _h4> ghz;
extern LookupFunc<float, float, sz1d, _h4> ghz_dr;

// generator of power potential
LookupFunc<float, float, sz1d, _h4> Fg_HertzianSq(float power);
LookupFunc<float, float, sz1d, _h4> Fg_HertzianSqDR(float power);