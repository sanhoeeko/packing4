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
    return global->rod->potential({ x,y,t });
}

float precisePotential(float x, float y, float t)
{
    return global->rod->HertzianRodPotential({ x,y,t });
}
