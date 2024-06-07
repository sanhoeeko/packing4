#pragma once

#include "defs.h"
#include "functional.h"

template<HowToCalGradient how> 
XytPair singleGradient(ParticlePair& ijxytt);


template<HowToCalGradient how>
void calGradient(PairInfo* pinfo, GradientBuffer* ge) {
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
            XytPair f = singleGradient<how>(src[i]);
            ptr[ii] -= f.first;
            ptr[jj] -= f.second;
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
            if (src[i].id2 == -114514) {
                // outside the boundary, add a penalty
                float h = src[i].t1;
                float f = potentialDR<ScreenedCoulomb>(4.0f - h);                    // fr > 0
                float r = sqrtf(src[i].x * src[i].x + src[i].y * src[i].y);
                ptr[ii] += {f * src[i].x / r, f * src[i].y / r, 0};
            }
            else {
                XytPair f = singleGradient<how>(src[i]);
                ptr[ii] -= f.first;
            }
        }
    }
}

template<HowToCalGradient how> 
GradientBuffer* PairInfo::CalGradient()
{
    static CacheFunction<PairInfo, GradientBuffer> f(calGradient<how>, new GradientBuffer(N));
    return f(this);
}