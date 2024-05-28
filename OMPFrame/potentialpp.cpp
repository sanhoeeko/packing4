#include"pch.h"
#include"array.h"
#include"potential.h"

const float rod_d = 0.25;

float _hertzianSq(const float& x2) {
    return pow(2 - sqrt(x2), 2.5f);
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

float hertzianSq(float x2) {
    static auto f = FHertzianSq();
    if (x2 >= 4)return 0;
    return f(x2);
}

Rod::Rod(int n, float d) :n(n), rod_d(d) {
    a = 1 + (n - 1) / 2.0f * rod_d;
    b = 1;
    shift = -(n - 1) / 2.0f;
    fv = new ReaderFunc<xyt, float, szxyt>(HashXyt);
    gate = Gate(a, b);
}

float Rod::HertzianRodPotential(const xyt& q) {
    float v = 0;
    for (int k = 0; k < n; k++) {
        float x2 = -(shift + k) * rod_d;
        for (int l = 0; l < n; l++) {
            float
                z = -(shift + l) * rod_d,
                xij = q.x + z * fcos(q.t) + x2,
                yij = q.y + z * fsin(q.t),
                r2 = xij * xij + yij * yij;
            v += hertzianSq(r2);
        }
    }
    return v;
}

float Rod::hertzian_rod_01(const xyt& q) {
    /*
        q is in [0,1] x [0,1] x [0,1]
    */
    xyt qinv = gate.inverse(q);
    return HertzianRodPotential(qinv);
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
            fv->data[HashXyt(q)] = hertzian_rod_01(q);
        }
    }
    // fv->write("potential.dat");
}

float Rod::potential(const xyt& q) {
    /*
        the input of fv is in [0,1] x [0,1] x [0,1]
    */
    return (*fv)(gate.transform(q));
}