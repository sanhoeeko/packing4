#include"pch.h"
#include"gradient.h"
#include"potential.h"
#include"functional.h"
#include"graph.h"

const float trivial_contact_energy = 1e-3;

void xyt::operator+=(const xyt& o) {
    x += o.x; y += o.y; t += o.t;
}

void xyt::operator-=(const xyt& o) {
    x -= o.x; y -= o.y; t -= o.t;
}

PairInfo::PairInfo()
{
}

PairInfo::PairInfo(int N)
{
    this->N = N;
    for (int i = 0; i < CORES; i++) {
        info_pp[i].reserve(N * N / 2);
        info_pw[i].reserve(N);
    }
    g_buffer = Maybe<GradientBuffer*>(new GradientBuffer(N));
    e_buffer = Maybe<EnergyBuffer*>(new EnergyBuffer(N));
    graph = Maybe<Graph<neighbors>*>(new Graph<neighbors>(N));
}

void PairInfo::clear()
{
#pragma omp parallel for num_threads(CORES)
    for (int i = 0; i < CORES; i++) {
        info_pp[i].clear();
        info_pw[i].clear();
    }
    g_buffer.clear();
    e_buffer.clear();
    graph.clear();
    is_filtered = false;
}


struct RotVector
{
    float s, c;

    RotVector(float angle) {
        s = fsin(angle); c = fcos(angle);
    }

    void rot(float* ptr, float* dst) {
        float x = ptr[0], y = ptr[1];
        dst[0] = c * x - s * y;
        dst[1] = s * x + c * y;
    }
    void inv(float* ptr, float* dst) {
        float x = ptr[0], y = ptr[1];
        dst[0] = c * x + s * y;
        dst[1] = -s * x + c * y;
    }
};

inline static float crossProduct(float* r, float* f) {
    return r[0] * f[1] - r[1] * f[0];
}

template<>
XytPair singleGradient<Normal>(ParticlePair& ijxytt) {
    xyt temp;
    RotVector rv = RotVector(ijxytt.t2);
    rv.inv(&ijxytt.x, &temp.x);
    temp.t = ijxytt.t2 - ijxytt.t1;
    xyt gradient = global->rod->gradient(temp);
    rv.rot((float*)&gradient, (float*)&gradient);
    float moment2 = -crossProduct(&ijxytt.x, (float*)&gradient) - gradient.t;   // parallel axis theorem !!
    return { gradient, {-gradient.x, -gradient.y, moment2} };
}

template<>
XytPair singleGradient<Test>(ParticlePair& ijxytt) {
    return global->rod->StandardGradient<ScreenedCoulomb>(ijxytt.x, ijxytt.y, ijxytt.t1, ijxytt.t2);
}

template<>
XytPair singleGradient<AsDisks>(ParticlePair& ijxytt) {
    float r2 = ijxytt.x * ijxytt.x + ijxytt.y * ijxytt.y;
    float fr = potentialDR<ScreenedCoulomb>(r2);
    float fx = fr * ijxytt.x, fy = fr * ijxytt.y;
    return { {fx, fy, 0}, {-fx, -fy, 0} };
}

inline static float singleEnergy(ParticlePair& ijxytt) {
    xyt temp;
    RotVector(ijxytt.t2).inv(&ijxytt.x, &temp.x);
    temp.t = ijxytt.t2 - ijxytt.t1;
    return global->rod->potential(temp);
}

void calEnergy(PairInfo* pinfo, EnergyBuffer* ge) {
    ge->clear();

    // for pp
#pragma omp parallel num_threads(CORES)
    {
        int idx = omp_get_thread_num();
        int n = pinfo->info_pp[idx].size();
        ParticlePair* src = pinfo->info_pp[idx].data();
        for (int i = 0; i < n; i++) {
            ge->buffers[idx] += singleEnergy(src[i]);
        }
    }
    // for pw 
#pragma omp parallel num_threads(CORES)
    {
        int idx = omp_get_thread_num();
        int n = pinfo->info_pw[idx].size();
        ParticlePair* src = pinfo->info_pw[idx].data();
        for (int i = 0; i < n; i++) {
            if (src[i].id2 != -114514) {
                ge->buffers[idx] += singleEnergy(src[i]) * 0.5f;
            }
        }
    }
}

EnergyBuffer* PairInfo::CalEnergy()
{
    if (!e_buffer.valid) {
        e_buffer.valid = true;
        calEnergy(this, e_buffer.obj);
    }
    return e_buffer.obj;
}

struct RodContactCondition {
    SegmentDist segment_dist;
    float r;
    RodContactCondition(float gamma) : r(2 / gamma), segment_dist(1 - 1 / gamma) {};
    bool operator()(ParticlePair& p) {
        return segment_dist(p) < r;
    }
};

PairInfo* PairInfo::filterAsRods(float gamma)
{
    if (!is_filtered) {
        is_filtered = true;
        RodContactCondition condition(gamma);
    #pragma omp parallel num_threads(CORES)
        {
            int idx = omp_get_thread_num();
            filter(info_pp[idx], condition);
        }
    }
    return this;
}

template<bool checkId2>
int contactNumber(vector<ParticlePair>* vp, float gamma) {
    int z[CORES] = { 0 };
    int sum_z = 0;
    RodContactCondition condition(gamma);
#pragma omp parallel num_threads(CORES)
    {
        int idx = omp_get_thread_num();
        int n = vp[idx].size();
        ParticlePair* src = vp[idx].data();
        if constexpr (checkId2) {
            for (int i = 0; i < n; i++) {
                if (src[i].id2 != -114514) {
                    z[idx] += (int)condition(src[i]);
                }
            }
        }
        else {
            for (int i = 0; i < n; i++) {
                z[idx] += (int)condition(src[i]);
            }
        }
    }
    for (int i = 0; i < CORES; i++) {
        sum_z += z[i];
    }
    return sum_z;
}

int PairInfo::contactNumberZ(float gamma)
{
    return contactNumber<false>(info_pp, gamma) + contactNumber<true>(info_pw, gamma);
}

inline static float dist(ParticlePair& ijxytt) {
    return sqrt(ijxytt.x * ijxytt.x + ijxytt.y * ijxytt.y);
}

float PairInfo::meanDistance(float gamma)
{
    int z[CORES] = { 0 };
    float d[CORES] = { 0.0f };
    int sum_z = 0;
    float sum_d = 0.0f;
    RodContactCondition condition(gamma);
#pragma omp parallel num_threads(CORES)
    {
        int idx = omp_get_thread_num();
        int n = info_pp[idx].size();
        ParticlePair* src = info_pp[idx].data();
        for (int i = 0; i < n; i++) {
            if (condition(src[i])) {
                z[idx]++;
                d[idx] += dist(src[i]);
            }
        }
    }
    for (int i = 0; i < CORES; i++) {
        sum_z += z[i];
        sum_d += d[i];
    }
    return sum_d / sum_z;
}

void getGraph(PairInfo* pinfo, Graph<neighbors>* graph) {
    // cannot use omp here, because `Graph<...>` is not a multithread container.
    for (int idx = 0; idx < CORES; idx++) {
        for (auto& pair : pinfo->info_pp[idx]) {
            graph->add_pair(pair.id1, pair.id2);
        }
    }
}

Graph<neighbors>* PairInfo::toGraph(float gamma)
{
    if (!graph.valid) {
        graph.valid = true;
        getGraph(filterAsRods(gamma), graph.obj);
    }
    return graph.obj;
}

GradientBuffer::GradientBuffer()
{
}

GradientBuffer::GradientBuffer(int N)
{
    this->N = N;
    for (int i = 0; i < CORES; i++) {
        buffers[i] = VectorXf::Zero(dof * N);
    }
    result = Maybe<VectorXf*>(new VectorXf(dof * N));
}

void GradientBuffer::clear()
{
#pragma omp parallel for num_threads(CORES)
    for (int i = 0; i < CORES; i++) {
        buffers[i].setZero();
    }
    result.valid = false;
    result.obj->setZero();
}

VectorXf GradientBuffer::join()
{
    if (!result.valid) {
        result.valid = true;
        if constexpr (CORES > 1)
        {
            for (int i = 0; i < CORES; i++) {
                *result.obj += buffers[i];
            }
        }
        else
        {
            *result.obj = buffers[0];     // one core specialization
        }
    }
    return *result.obj;
}

EnergyBuffer::EnergyBuffer()
{
}

EnergyBuffer::EnergyBuffer(int N)
{
    this->N = N;
    clear();
}

void EnergyBuffer::clear()
{
    for (int i = 0; i < CORES; i++) {
        buffers[i] = 0;
    }
    result.valid = false;
    result.obj = 0;
}

float EnergyBuffer::sum()
{
    if (!result.valid) {
        result.valid = true;
        for (int i = 0; i < CORES; i++) {
            result.obj += buffers[i];
        }
    }
    return result.obj;
}
