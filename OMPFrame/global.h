#pragma once

#include "potential.h"

struct Global {
    Rod* rod;
};

void setGlobal();

void setRod(int n, float d);

float fastPotential(float x, float y, float t);

float interpolatePotential(float x, float y, float t);

float precisePotential(float x, float y, float t);