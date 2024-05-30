#include"pch.h"
#include"gradient.h"
#include"potential.h"
#include"functional.h"

PairInfo::PairInfo(int N)
{
    this->id = -1;
    this->N = N;
    for (int i = 0; i < CORES; i++) {
        info[i].reserve(N);
    }
}

void PairInfo::clear()
{
    for (int i = 0; i < CORES; i++) {
        info[i].clear();
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

#pragma omp parallel
    {
        int idx = omp_get_thread_num();
        int n = pinfo->info[idx].size();
        xyt* ptr = (xyt*)(void*)ge->buffers[idx].data();
        for (int i = 0; i < n; i++) {
            int
                ii = pinfo->info[idx][i].id1, 
                jj = pinfo->info[idx][i].id2;
            xyt f = singleGradient(pinfo->info[idx][i]);
            ptr[ii] -= f;
            ptr[jj] += f;
        }
    }
}

void calGradientAsDisks(PairInfo* pinfo, GradientAndEnergy* ge) {
    ge->clear();

#pragma omp parallel
    {
        int idx = omp_get_thread_num();
        int n = pinfo->info[idx].size();
        xyt* ptr = (xyt*)(void*)ge->buffers[idx].data();
        for (int i = 0; i < n; i++) {
            int
                ii = pinfo->info[idx][i].id1,
                jj = pinfo->info[idx][i].id2;
            xyt f = singleGradientAsDisks(pinfo->info[idx][i]);
            ptr[ii] -= f;
            ptr[jj] += f;
        }
    }
}

void calEnergyAsDisks(PairInfo* pinfo, GradientAndEnergy* ge) {

#pragma omp parallel
    {
        int idx = omp_get_thread_num();
        int n = pinfo->info[idx].size();
        xyt* ptr = (xyt*)(void*)ge->buffers[idx].data();
        for (int i = 0; i < n; i++) {
            int
                ii = pinfo->info[idx][i].id1,
                jj = pinfo->info[idx][i].id2;
            xyt f = singleGradientAsDisks(pinfo->info[idx][i]);
            ptr[ii] -= f;
            ptr[jj] += f;
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
}
