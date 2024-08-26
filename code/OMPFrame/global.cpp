#include "pch.h"
#include "global.h"
#include "gradient.h"
#include "analysis.h"
#include "optimizer.h"

Global* global;

const char* potential_file_name = "potential.dat";

void setGlobal()
{
    global = new Global();
    global->pf = PotentialFunc(-1);
    global->power_of_potential = 0;
}

void setEnums(int potential_func)
{
    global->pf = (PotentialFunc)potential_func;
}

void setPotentialPower(float power)
{
    if (power != global->power_of_potential) {
        // generate scalar function look-up table
        ghz = Fg_HertzianSq(power);
        ghz_dr = Fg_HertzianSqDR(power);
    }
    global->power_of_potential = power;
}

void declareRod(int n, float d) {
    global->rod = new Rod(n, d);
}

void setRod(int n, float d, int threads)
{
    typedef void (Rod::*Func)(int);
    static Func funcs[2] = {
        &Rod::initPotential<Hertzian>,
        &Rod::initPotential<ScreenedCoulomb>,
    };

    declareRod(n, d);
    global->checkPowerOfPotential();             // crash if the power of potential is undefined
    (global->rod->*funcs[global->pf])(threads);  // initPotential: calculate the potential table
}

void* createState(int N, float boundary_a, float boundary_b)
{ 
    auto state = global->newState(N);
    state->setBoundary(boundary_a, boundary_b);
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

float meanDistance(void* state_ptr, float gamma)
{
    State* s = reinterpret_cast<State*>(state_ptr);
    return s->meanDistance(gamma);
}

float meanContactZ(void* state_ptr, float gamma)
{
    State* s = reinterpret_cast<State*>(state_ptr);
    return s->meanContactZ(gamma);
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

void* loadState(void* data_src, int N, float boundary_a, float boundary_b)
{
    State* s = global->newState(N);
    s->setBoundary(boundary_a, boundary_b);
    s->loadFromData((float*)data_src);
    return s;
}

void freeState(void* state_ptr)
{
    global->states.free(reinterpret_cast<State*>(state_ptr));
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

template<HowToCalGradient how>
float* landscapeAlongGradient_helper(void* state_ptr, float max_stepsize, int samples)
{
    static float* res = NULL;
    if (res) {
        delete[] res; res = NULL;
    }
    res = new float[samples];
    State* s = reinterpret_cast<State*>(state_ptr);
    vector<float> energies = _landscapeAlongGradient<how>(s, max_stepsize, samples);
    memcpy(res, energies.data(), samples * sizeof(float));
    return res;
}

float* landscapeAlongGradient(void* state_ptr, float max_stepsize, int samples) 
{
    return landscapeAlongGradient_helper<Normal>(state_ptr, max_stepsize, samples);
}

float* landscapeLBFGS(void* state_ptr, float max_stepsize, int samples)
{
    return landscapeAlongGradient_helper<LBFGS>(state_ptr, max_stepsize, samples);
}

float* landscapeOnGradientSections(void* state_ptr, float max_stepsize, int samples)
{
    static float* res = NULL;
    if (res) {
        delete[] res; res = NULL;
    }
    res = new float[(2 * samples + 1) * samples];
    State* s = reinterpret_cast<State*>(state_ptr);
    vector<vector<float>> energies = _landscapeOnGradientSections(s, max_stepsize, samples);
    for (int i = 0; i < 2 * samples + 1; i++) {
        memcpy(res + i * samples, energies[i].data(), samples * sizeof(float));
    }
    return res;
}

float meanS(void* state_ptr, float gamma)
{
    State* s = reinterpret_cast<State*>(state_ptr);
    return s->orderS_ave(gamma);
}

float absPhi(void* state_ptr, float gamma, int p)
{
    State* s = reinterpret_cast<State*>(state_ptr);
    return std::abs(s->orderPhi_ave(gamma, p));
}

float* Si(void* state_ptr, float gamma)
{
    static float* res = NULL;
    State* s = reinterpret_cast<State*>(state_ptr);
    if (res) {
        delete[] res; res = NULL;
    }
    res = new float[s->N];
    VectorXf si = s->orderS(gamma);
    memcpy(res, si.data(), s->N * sizeof(float));
    return res;
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

float eqLineGD(void* state_ptr, int max_iterations)
{
    State* s = reinterpret_cast<State*>(state_ptr);
    return s->eqLineGD(max_iterations);
}

float eqLBFGS(void* state_ptr, int max_iterations)
{
    State* s = reinterpret_cast<State*>(state_ptr);
    return s->eqLBFGS(max_iterations);
}

float eqMix(void* state_ptr, int max_iterations)
{
    State* s = reinterpret_cast<State*>(state_ptr);
    return s->eqMix(max_iterations);
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
    static Func funcs[PotentialFunc_Count] = {
        &Rod::StandardPotential<Hertzian>,
        &Rod::StandardPotential<ScreenedCoulomb>,
        &Rod::StandardPotential<GeneralizedHertzian>
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
    static Func funcs[PotentialFunc_Count] = {
        &Rod::StandardGradient<Hertzian>,
        &Rod::StandardGradient<ScreenedCoulomb>,
        &Rod::StandardGradient<GeneralizedHertzian>
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

float testERoot(void* state_ptr, float max_stepsize)
{
    State* s = reinterpret_cast<State*>(state_ptr);
    VectorXf g = normalize(s->CalGradient<Normal>());
    return ERoot(s, g, max_stepsize);
}

float testBestStepSize(void* state_ptr, float max_stepsize)
{
    try {
        State* s = reinterpret_cast<State*>(state_ptr);
        VectorXf g = normalize(s->CalGradient<Normal>());
        return BestStepSize(s, g, max_stepsize);
    }
    catch (int exception) {
        return NAN;
    }
}
