#include"pch.h"
#include"gradient.h"
#include"potential.h"
#include"functional.h"

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

xyt singleGradient(ParticlePair& ijxytt) {
    Map<Vector2f> xy(&ijxytt.x);
    Vector2f xy_rotated = FU(-ijxytt.t1) * xy;
    // xyt gradient = interpolateGradientSimplex(xy_rotated[0], xy_rotated[1], ijxytt.t2 - ijxytt.t1);
    xyt gradient = { 0 };
    Map<Vector2f> force_rotated((float*)&gradient);
    force_rotated = FU(ijxytt.t1) * force_rotated;      // mul inplace
    return gradient;
}

xyt singleGradientAsDisks(ParticlePair& ijxytt) {
    float r2 = ijxytt.x * ijxytt.x + ijxytt.y * ijxytt.y;
    float f_r = d_isotropicSq_r(r2);
    return { f_r * ijxytt.x, f_r * ijxytt.y, 0 };
}

void xyt::operator+=(const xyt& o) {
    x += o.x; y += o.y; t += o.t;
}

void xyt::operator-=(const xyt& o){
    x -= o.x; y -= o.y; t -= o.t;
}

void calGradient(PairInfo* pinfo, GradientAndEnergy* ge) {
    ge->clear();

    // for pp
#pragma omp parallel num_threads(CORES)
    {
        int idx = omp_get_thread_num();
        int n = pinfo->info_pp[idx].size();
        xyt* ptr = (xyt*)(void*)ge->buffers[idx].data();
        for (int i = 0; i < n; i++) {
            int
                ii = pinfo->info_pp[idx][i].id1,
                jj = pinfo->info_pp[idx][i].id2;
            xyt f = singleGradient(pinfo->info_pp[idx][i]);
            ptr[ii] -= f;
            ptr[jj] += f;
        }
    }
    // for pw
#pragma omp parallel num_threads(CORES)
    {
        int idx = omp_get_thread_num();
        int n = pinfo->info_pw[idx].size();
        xyt* ptr = (xyt*)(void*)ge->buffers[idx].data();
        for (int i = 0; i < n; i++) {
            int ii = pinfo->info_pw[idx][i].id1;                        // jj = -1
            xyt f = singleGradient(pinfo->info_pw[idx][i]);
            ptr[ii] -= f;
        }
    }
}

void calGradientAsDisks(PairInfo* pinfo, GradientAndEnergy* ge) {
    ge->clear();

    // for pp
#pragma omp parallel num_threads(CORES)
    {
        int idx = omp_get_thread_num();
        int n = pinfo->info_pp[idx].size();
        xyt* ptr = (xyt*)(void*)ge->buffers[idx].data();
        ParticlePair* src = (ParticlePair*)(void*)pinfo->info_pp[idx].data();
        for (int i = 0; i < n; i++) {
            int
                ii = src[i].id1,
                jj = src[i].id2;
            xyt f = singleGradientAsDisks(src[i]);
            ptr[ii] -= f;
            ptr[jj] += f;
        }
    }
    // for pw 
#pragma omp parallel num_threads(CORES)
    {
        int idx = omp_get_thread_num();
        int n = pinfo->info_pw[idx].size();
        xyt* ptr = (xyt*)(void*)ge->buffers[idx].data();
        ParticlePair* src = (ParticlePair*)(void*)pinfo->info_pw[idx].data();
        for (int i = 0; i < n; i++) {
            int ii = src[i].id1;
            xyt f = singleGradientAsDisks(src[i]);
            ptr[ii] -= f;
        }
    }
}

GradientAndEnergy* PairInfo::CalGradient()
{
    static CacheFunction<PairInfo, GradientAndEnergy> f(calGradient, new GradientAndEnergy(N));
    return f(this);
}

GradientAndEnergy* PairInfo::CalGradientAsDisks()
{
    static CacheFunction<PairInfo, GradientAndEnergy> f(calGradientAsDisks, new GradientAndEnergy(N));
    return f(this);
}

GradientAndEnergy* PairInfo::CalEnergy()
{
    return nullptr;
}

GradientAndEnergy::GradientAndEnergy(int N)
{
    this->id = -1;
    this->N = N;
    for (int i = 0; i < CORES; i++) {
        buffers[i] = VectorXf::Zero(3 * N);
    }
}

void GradientAndEnergy::clear()
{
    for (int i = 0; i < CORES; i++) {
        buffers[i].setZero();
    }
}

void GradientAndEnergy::joinTo(VectorXf* g)
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
