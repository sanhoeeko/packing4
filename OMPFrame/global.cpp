#include "pch.h"
#include "global.h"

Global* global;

const char* potential_file_name = "potential.dat";

void setGlobal()
{
    global = new Global();
    global->pf = PotentialFunc(-1);
}

void setEnums(int potential_func)
{
    global->pf = (PotentialFunc)potential_func;
}

void setRod(int n, float d)
{
    typedef void (Rod::*Func)(void);
    Func funcs[2] = {
        &Rod::initPotential<Hertzian>,
        &Rod::initPotential<ScreenedCoulomb>,
    };

    global->rod = new Rod(n, d);
    (global->rod->*funcs[global->pf])();
}

void* createState(int N, float boundary_a, float boundary_b)
{ 
    auto state = new State(N);
    state->boundary = new EllipseBoundary(boundary_a, boundary_b);
    state->randomInitStateCC();
    return state;
}

void* getStateData(void* state_ptr)
{
    State* s = reinterpret_cast<State*>(state_ptr);
    return s->configuration.data();
}

int getStateIterations(void* state_ptr)
{
    State* s = reinterpret_cast<State*>(state_ptr);
    return s->max_gradient_amps.size();
}

void* getStateResidualForce(void* state_ptr)
{
    State* s = reinterpret_cast<State*>(state_ptr);
    return s->CalGradient<AsDisks>()->data();
}

void readPotential(int n, float d)
{
    global->rod = new Rod(n, d);
    global->rod->fv->read(potential_file_name);
}

void writePotential()
{
    global->rod->fv->write(potential_file_name);
}

int getPotentialId()
{
    return global->pf;
}

void* getStateMaxGradients(void* state_ptr)
{
    State* s = reinterpret_cast<State*>(state_ptr);
    return s->max_gradient_amps.data();
}

void initStateAsDisks(void* state_ptr) {
    State* s = reinterpret_cast<State*>(state_ptr);
    return s->initAsDisks();
}

void setBoundary(void* state_ptr, float boundary_a, float boundary_b)
{
    State* s = reinterpret_cast<State*>(state_ptr);
    s->setBoundary(boundary_a, boundary_b);
}

void equilibriumGD(void* state_ptr)
{
    State* s = reinterpret_cast<State*>(state_ptr);
    s->equilibriumGD();
}

float fastPotential(float x, float y, float t)
{
    return global->rod->potentialNoInterpolate({ x,y,t });
}

float interpolatePotential(float x, float y, float t)
{
    if (abs(x) > 2 * global->rod->a || abs(y) > global->rod->a + global->rod->b) {
        return 0;
    }
    else {
        return global->rod->potential({ x,y,t });
    }
}

float precisePotential(float x, float y, float t)
{
    typedef float (Rod::* Func)(const xyt&);
    Func funcs[2] = {
        &Rod::StandardPotential<Hertzian>,
        &Rod::StandardPotential<ScreenedCoulomb>
    };

    return (global->rod->*funcs[global->pf])({
        abs(x),
        abs(y),
        (x > 0) ^ (y > 0) ? t : pi - t,
    });
}

float* interpolateGradient(float x, float y, float t)
{
    static float arr[3];
    if (abs(x) > 2 * global->rod->a || abs(y) > global->rod->a + global->rod->b) {
        arr[0] = 0; arr[1] = 0; arr[2] = 0;
    }
    else {
        xyt g = global->rod->gradient({ x,y,t });
        arr[0] = g.x; arr[1] = g.y; arr[2] = g.t;
    }
    return arr;
}

float* gradientReference(float x, float y, float t1, float t2)
{
    typedef XytPair(Rod::* Func)(float, float, float, float);
    static Func funcs[2] = {
        &Rod::StandardGradient<Hertzian>,
        &Rod::StandardGradient<ScreenedCoulomb>,
    };
    static float arr[6];
    XytPair g = (global->rod->*funcs[global->pf])(x, y, t1, t2);
    memcpy(arr, &g, sizeof(XytPair));
    return arr;
}

float* gradientTest(float x, float y, float t1, float t2)
{
    static float arr[6];
    if (x * x + y * y > 4) {
        memset(arr, 0, sizeof(XytPair));
    }
    else {
        ParticlePair pp = { 0, 0, x, y, t1, t2 };
        XytPair g = singleGradient<Normal>(pp);
        memcpy(arr, &g, sizeof(XytPair));
    }
    return arr;
}

float* getMirrorOf(float A, float B, float x, float y, float t)
{
    static float arr[3];
    EllipseBoundary b = EllipseBoundary(A, B);
    Maybe<ParticlePair> pp = b.collide(0, { x,y,t });
    if (!pp.valid || pp.obj.id2 == -114514) {
        memset(arr, 0, 3 * sizeof(float));
    }
    else {
        float
            dx = pp.obj.x,
            dy = pp.obj.y;
        arr[0] = x - dx;
        arr[1] = y - dy;
        arr[2] = pp.obj.t2;
    }
    return arr;
}
