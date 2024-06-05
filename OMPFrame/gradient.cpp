#include"pch.h"
#include"gradient.h"
#include"potential.h"
#include"functional.h"

void xyt::operator+=(const xyt& o) {
    x += o.x; y += o.y; t += o.t;
}

void xyt::operator-=(const xyt& o) {
    x -= o.x; y -= o.y; t -= o.t;
}

bool isDataValid(VectorXf* q) {
    static bool bs[CORES];
    memset(bs, 1, CORES * sizeof(bool));
    int stride = q->size() / CORES;
    float* ptr = q->data();
#pragma omp parallel num_threads(CORES)
    {
        int idx = omp_get_thread_num();
        int end = idx + 1 == CORES ? q->size() : (idx + 1) * stride;
        for (int i = idx * stride; i < end; i++) {
            if (isnan(ptr[i])) {
                bs[idx] = false; break;
            }
        }
    }
    for (int i = 0; i < CORES; i++) {
        if (!bs[i])return false;
    }
    return true;
}

PairInfo::PairInfo(int N)
{
    this->id = -1;
    this->N = N;
    for (int i = 0; i < CORES; i++) {
        info_pp[i].reserve(N);
        info_pw[i].reserve(N);
    }
}

void PairInfo::clear()
{
    for (int i = 0; i < CORES; i++) {
        info_pp[i].clear();
        info_pw[i].clear();
    }
}


inline void rotVector(float angle, float* ptr, float* dst) {
    /*
        when `ptr == dst`, it is an inplace rotation
    */
    float
        s = fsin(angle),
        c = fcos(angle),
        x = ptr[0],
        y = ptr[1];
    dst[0] = c * x - s * y;
    dst[1] = s * x + c * y;
}

inline float crossProduct(float* r, float* f) {
    return r[0] * f[1] - r[1] * f[0];
}

template<>
XytPair singleGradient<Normal>(ParticlePair& ijxytt) {
    float theta = ijxytt.t2;
    xyt temp;
    rotVector(-theta, &ijxytt.x, &temp.x);
    temp.t = ijxytt.t2 - ijxytt.t1;
    xyt gradient = global->rod->gradient(temp);
    rotVector(theta, (float*)&gradient, (float*)&gradient);
    float moment2 = -crossProduct(&ijxytt.x, (float*)&gradient) - gradient.t;   // parallel axis theorem !!
    return { gradient, {-gradient.x, -gradient.y, moment2} };
}

template<>
XytPair singleGradient<AsDisks>(ParticlePair& ijxytt) {
    float r2 = ijxytt.x * ijxytt.x + ijxytt.y * ijxytt.y;
    float fr = d_isotropicSq_r(r2);
    float fx = fr * ijxytt.x, fy = fr * ijxytt.y;
    return { {fx, fy, 0}, {-fx, -fy, 0} };
}

float singleEnergy(ParticlePair& ijxytt) {
    Map<Vector2f> xy(&ijxytt.x);
    Vector2f xy_rotated = FU(-ijxytt.t1) * xy;
    return global->rod->potential({ xy_rotated[0], xy_rotated[1], ijxytt.t2 - ijxytt.t1 });
}

void calEnergy(PairInfo* pinfo, EnergyBuffer* ge) {
    ge->clear();

    // for pp
#pragma omp parallel num_threads(CORES)
    {
        int idx = omp_get_thread_num();
        int n = pinfo->info_pp[idx].size();
        ParticlePair* src = (ParticlePair*)(void*)pinfo->info_pp[idx].data();
        for (int i = 0; i < n; i++) {
            ge->buffers[idx] += singleEnergy(src[i]);
        }
    }
    // for pw 
#pragma omp parallel num_threads(CORES)
    {
        int idx = omp_get_thread_num();
        int n = pinfo->info_pw[idx].size();
        ParticlePair* src = (ParticlePair*)(void*)pinfo->info_pw[idx].data();
        for (int i = 0; i < n; i++) {
            if (src[i].id2 != 114514) {
                ge->buffers[idx] += singleEnergy(src[i]) * 0.5f;
            }
        }
    }
}

EnergyBuffer* PairInfo::CalEnergy()
{
    static CacheFunction<PairInfo, EnergyBuffer> f(calEnergy, new EnergyBuffer(N));
    return f(this);
}

GradientBuffer::GradientBuffer(int N)
{
    this->id = -1;
    this->N = N;
    for (int i = 0; i < CORES; i++) {
        buffers[i] = VectorXf::Zero(3 * N);
    }
}

void GradientBuffer::clear()
{
    for (int i = 0; i < CORES; i++) {
        buffers[i].setZero();
    }
}

void GradientBuffer::joinTo(VectorXf* g)
{
    g->setZero();
    for (int i = 0; i < CORES; i++) {
        *g += buffers[i];
    }
#if ENABLE_NAN_CHECK
    if (!isDataValid(g)) {
        cout << "nan in gradient. checked in `joinTo`" << endl;
        throw 114514;
    }
#endif
}

EnergyBuffer::EnergyBuffer(int N)
{
    this->id = -1;
    this->N = N;
    clear();
}

void EnergyBuffer::clear()
{
    for (int i = 0; i < CORES; i++) {
        buffers[i] = 0;
    }
}

float EnergyBuffer::sum()
{
    float s = 0;
    for (int i = 0; i < CORES; i++) {
        s += buffers[i];
    }
    return s;
}