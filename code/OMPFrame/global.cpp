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

void setRod(int n, float d, int threads)
{
    typedef void (Rod::*Func)(int);
    static Func funcs[2] = {
        &Rod::initPotential<Hertzian>,
        &Rod::initPotential<ScreenedCoulomb>,
    };

    global->rod = new Rod(n, d);
    (global->rod->*funcs[global->pf])(threads);
}

void* createState(int N, float boundary_a, float boundary_b)
{ 
    auto state = global->newState(N);
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
    return s->ge.size();
}

void* getStateResidualForce(void* state_ptr)
{
    static VectorXf res;
    State* s = reinterpret_cast<State*>(state_ptr);
    res = s->CalGradient<Normal>();
    return res.data();
}

float getStateMaxResidualForce(void* state_ptr)
{
    static VectorXf temp;
    State* s = reinterpret_cast<State*>(state_ptr);
    VectorXf g = VectorXf::Zero(s->N);

    temp = s->CalGradient<Normal>();
    xyt* ptr = (xyt*)temp.data();
    
#pragma omp parallel for num_threads(CORES)
    for (int i = 0; i < s->N; i++) {
        g[i] = ptr[i].amp2();
    }
    return sqrt(g.maxCoeff());
}

int getSiblingId(void* state_ptr)
{
    State* s = reinterpret_cast<State*>(state_ptr);
    return s->sibling_id;
}

void setStateData(void* state_ptr, void* data_src)
{
    State* s = reinterpret_cast<State*>(state_ptr);
    s->loadFromData((float*)data_src);
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

float* landscapeAlongGradient(void* state_ptr, float max_stepsize, int samples)
{
    static vector<float>* res = new vector<float>();
    State* s = reinterpret_cast<State*>(state_ptr);
    vector<float> energies = _landscapeAlongGradient(s, max_stepsize, samples);
    res->resize(energies.size());
    memcpy(res->data(), energies.data(), energies.size() * sizeof(float));
    return res->data();
}

void* getStateMaxGradOrEnergy(void* state_ptr)
{
    State* s = reinterpret_cast<State*>(state_ptr);
    return s->ge.data();
}

void initStateAsDisks(void* state_ptr) {
    State* s = reinterpret_cast<State*>(state_ptr);
    s->initAsDisks((int)1e5);
}

void setBoundary(void* state_ptr, float boundary_a, float boundary_b)
{
    State* s = reinterpret_cast<State*>(state_ptr);
    s->setBoundary(boundary_a, boundary_b);
}

void singleStep(void* state_ptr, int mode, float step_size)
{
    typedef VectorXf (State::* Func)();
    static Func funcs[HowToCalGradient_Count] = {
        &State::CalGradient<Normal>,
        &State::CalGradient<AsDisks>,
    };

    State* s = reinterpret_cast<State*>(state_ptr);
    VectorXf g = (s->*funcs[mode])();
    s->descent(step_size, g);
}

float equilibriumGD(void* state_ptr, int max_iterations)
{
    State* s = reinterpret_cast<State*>(state_ptr);
    return s->equilibriumGD(max_iterations);
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
    static Func funcs[2] = {
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
    // note: this function is single threaded
    static float arr[dof];
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
    // note: this function is single threaded
    typedef XytPair(Rod::* Func)(float, float, float, float);
    static Func funcs[2] = {
        &Rod::StandardGradient<Hertzian>,
        &Rod::StandardGradient<ScreenedCoulomb>,
    };
    static float arr[2 * dof];
    XytPair g = (global->rod->*funcs[global->pf])(x, y, t1, t2);
    memcpy(arr, &g, sizeof(XytPair));
    return arr;
}

float* gradientTest(float x, float y, float t1, float t2)
{
    // note: this function is single threaded
    static float arr[2 * dof];
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
    // note: this function is single threaded
    static float arr[dof];
    EllipseBoundary b = EllipseBoundary(A, B);
    Maybe<ParticlePair> pp = b.collide(0, { x,y,t });
    if (!pp.valid || pp.obj.id2 == -114514) {
        memset(arr, 0, dof * sizeof(float));
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

State* Global::newState(int N)
{
    State* state = new State(N, global->states.size());
    global->states.push_back(state);
    return state;
}
