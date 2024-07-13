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

PairInfo::PairInfo()
{
}

PairInfo::PairInfo(int N)
{
    this->N = N;
    for (int i = 0; i < CORES; i++) {
        info_pp[i].reserve(N);
        info_pw[i].reserve(N);
    }
    g_buffer = Maybe<GradientBuffer*>(new GradientBuffer(N));
    e_buffer = Maybe<EnergyBuffer*>(new EnergyBuffer(N));
}

void PairInfo::clear()
{
    for (int i = 0; i < CORES; i++) {
        info_pp[i].clear();
        info_pw[i].clear();
    }
    g_buffer.clear();
    e_buffer.clear();
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

inline float crossProduct(float* r, float* f) {
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

float singleEnergy(ParticlePair& ijxytt) {
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