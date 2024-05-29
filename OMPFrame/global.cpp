#include "pch.h"
#include "global.h"

Global* global;

void setGlobal()
{
    global = new Global();
}

void setRod(int n, float d)
{
    global->rod = new Rod(n, d);
    global->rod->initPotential();
}

float fastPotential(float x, float y, float t)
{
    return global->rod->potentialNoInterpolate({ x,y,t });
}

float interpolatePotential(float x, float y, float t)
{
    return global->rod->potential({ x,y,t });
}

float precisePotential(float x, float y, float t)
{
    return global->rod->HertzianRodPotential({
        abs(x),
        abs(y),
        (x > 0) ^ (y > 0) ? t : pi - t,
    });
}
