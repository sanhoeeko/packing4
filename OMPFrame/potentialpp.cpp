#include"pch.h"
#include"array.h"
#include"potential.h"

float _hertzianSq(const float& x2) {
    return pow(2 - sqrt(x2), 2.5f);
}

float _d_isotropicSq_r(const float& x2) {
    if (x2 == 0)return 0;
    float x = sqrt(x2);
    return (exp(-x) / x) / x;
}

template<size_t capacity>
size_t hash04(const float& x) {
    return x * (capacity / 4);
}

static ReaderFunc<float, float, sz1d> FHertzianSq() {
    static vector<float> xs = linspace(0, 4, sz1d);
    static auto f = new ReaderFunc<float, float, sz1d>(_hertzianSq, hash04<sz1d>, xs);
    return *f;
}

static ReaderFunc<float, float, sz1d> FIsotropicSqDR() {
    static vector<float> xs = linspace(0, 4, sz1d);
    static auto f = new ReaderFunc<float, float, sz1d>(_d_isotropicSq_r, hash04<sz1d>, xs);
    return *f;
}

float hertzianSq(float x2) {
    static auto f = FHertzianSq();
    if (x2 >= 4)return 0;
    return f(x2);
}

float d_isotropicSq_r(float x2) {
    static auto f = FIsotropicSqDR();
    if (x2 >= 4)return 0;
    return f(x2);
}

Rod::Rod(int n, float d) :n(n), rod_d(d) {
    a = 1 + (n - 1) / 2.0f * rod_d + 0.1f;   // 0.1 (zero padding) is for memory safe
    b = 1 + 0.1f;
    c = a - b;
    n_shift = -(n - 1) / 2.0f;
    fv = new ReaderFunc<xyt, float, szxyt>(HashXyt);
}

/*
    The origin definition of the potential
*/
float Rod::HertzianRodPotential(const xyt& q) {
#if NAN_IF_PENETRATE
    if (isSegmentCrossing(q)) return NAN;
#endif
    float v = 0;
    for (int k = 0; k < n; k++) {
        float xpos = (n_shift + k) * rod_d;
        for (int l = 0; l < n; l++) {
            float
                z = (n_shift + l) * rod_d,
                xij = q.x + z * fcos(q.t) + xpos,
                yij = q.y + z * fsin(q.t),
                r2 = xij * xij + yij * yij;
            v += hertzianSq(r2);
        }
    }
    return v;
}

void Rod::initPotential() {
    vector<float> xs = linspace_including_endpoint(0, 1, szx);
    vector<float> ys = linspace_including_endpoint(0, 1, szy);
    vector<float> ts = linspace_including_endpoint(0, 1, szt);
    size_t m = xs.size();

#pragma omp parallel for num_threads(CORES)
    for (int i = 0; i < m; i++) {
        float x = xs[i];
        for (float y : ys) for (float t : ts) {
            xyt q = { x,y,t };
            fv->data[HashXyt(q)] = HertzianRodPotential(inverse(q));
        }
    }
    // fv->write("potential.dat");
}

float Rod::potentialNoInterpolate(const xyt& q)
{
    return (*fv)(transform(q));
}

float Rod::potential(const xyt& q) {
    /*
        q: real x y theta
    */
    return interpolatePotentialSimplex(transform(q));
}

xyt Rod::gradient(const xyt& q) {
    /*
        q: real x y theta
    */
    return interpolateGradientSimplex(transform(q));
}