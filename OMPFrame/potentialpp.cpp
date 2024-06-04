#include"pch.h"
#include"array.h"
#include"potential.h"

inline static XytPair ZeroXytPair() {
    return { 0,0,0,0,0,0 };
}

float _hertzianSq(const float& x2) {
    return pow(2 - sqrt(x2), 2.5f);
}

float _hertzianSqDR(const float& x2) {
    float x = sqrt(x2);
    return 2.5 * pow(2 - x, 1.5f) / x;
}

float _isotropicSqDR(const float& x2) {
    if (x2 == 0)return 0;
    float x = sqrt(x2);
    return -(exp(-x) / x) / x;
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

static ReaderFunc<float, float, sz1d> FHertzianSqDR() {
    static vector<float> xs = linspace(0, 4, sz1d);
    static auto f = new ReaderFunc<float, float, sz1d>(_hertzianSqDR, hash04<sz1d>, xs);
    return *f;
}

static ReaderFunc<float, float, sz1d> FIsotropicSqDR() {
    static vector<float> xs = linspace(0, 4, sz1d);
    static auto f = new ReaderFunc<float, float, sz1d>(_isotropicSqDR, hash04<sz1d>, xs);
    return *f;
}

float hertzianSq(float x2) {
    static auto f = FHertzianSq();
    if (x2 >= 4)return 0;
    return f(x2);
}

float hertzianSqDR(float x2) {
    static auto f = FHertzianSqDR();
    if (x2 >= 4)return 0;
    return f(x2);
}

float d_isotropicSq_r(float x2) {
    static auto f = FIsotropicSqDR();
    if (x2 >= 4)return 0;
    return f(x2);
}

Rod::Rod(int n, float d) {
    a = 1,
    b = 1 / (1 + (n - 1) * d / 2.0f),
    c = a - b;
    this->n = n;
    this->rod_d = d * b;
    a_padded = a + 0.01f;    // zero padding, for memory safety
    b_padded = b + 0.01f;
    this->n_shift = -(n - 1) / 2.0f;
    this->inv_disk_R2 = 1 / (b * b);
    this->fv = new ReaderFunc<xyt, float, szxyt>(HashXyt);
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
        float z1 = (n_shift + k) * rod_d;
        for (int l = 0; l < n; l++) {
            float z2 = (n_shift + l) * rod_d;
            float
                xij = q.x - z1 + z2 * fcos(q.t),
                yij = q.y + z2 * fsin(q.t),
                r2 = (xij * xij + yij * yij) * inv_disk_R2;
            v += hertzianSq(r2);
        }
    }
    return v;
}

XytPair Rod::HertzianGradientStandard(float x, float y, float t1, float t2){
#if NAN_IF_PENETRATE
    if (isSegmentCrossing(q)) return NAN;
#endif
    XytPair g = ZeroXytPair();
    for (int k = 0; k < n; k++) {
        float z1 = (n_shift + k) * rod_d;
        for (int l = 0; l < n; l++) {
            float z2 = (n_shift + l) * rod_d;
            float
                xij = x - z1 * fcos(t1) + z2 * fcos(t2),
                yij = y - z1 * fsin(t1) + z2 * fsin(t2),
                r2 = (xij * xij + yij * yij) * inv_disk_R2,
                fr = hertzianSqDR(r2);
            if (fr != 0.0f) {
                float
                    fx = fr * xij,
                    fy = fr * yij,
                    torque1 = z1 * (fx * fsin(t1) - fy * fcos(t1)),
                    torque2 = -z2 * (fx * fsin(t2) - fy * fcos(t2));
                g.first += {fx, fy, torque1};
                g.second += {-fx, -fy, torque2};
            }
        }
    }
    return g;
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
    xyt g = transform_signed(interpolateGradientSimplex(transform(q)));
    bool 
        sign_x = q.x < 0,
        sign_y = q.y < 0;
    if (sign_x)g.x = -g.x;
    if (sign_y)g.y = -g.y;
    if (sign_x ^ sign_y)g.t = -g.t;
    return g;
}