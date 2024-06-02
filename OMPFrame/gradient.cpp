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

/*
    The range of (x,y,t): x = X/(2a) in [0,1), y = Y/(a+b) in [0,1), t = Theta/pi in [0,1)
    if szx == szy == szz, the maximal szx is 1024 for the sake of size_t.
*/

template<>
xyt singleGradient<Normal>(ParticlePair& ijxytt) {
    Map<Vector2f> xy(&ijxytt.x);
    Vector2f xy_rotated = FU(-ijxytt.t1) * xy;
    xyt gradient = global->rod->gradient({ xy_rotated[0], xy_rotated[1], ijxytt.t2 - ijxytt.t1 });
    Map<Vector2f> force_rotated((float*)&gradient);
    force_rotated = FU(ijxytt.t1) * force_rotated;      // mul inplace
    return gradient;
}

template<>
xyt singleGradient<AsDisks>(ParticlePair& ijxytt) {
    float r2 = ijxytt.x * ijxytt.x + ijxytt.y * ijxytt.y;
    float f_r = d_isotropicSq_r(r2);
    return { f_r * ijxytt.x, f_r * ijxytt.y, 0 };
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
            ge->buffers[idx] += singleEnergy(src[i]) * 0.5f;
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