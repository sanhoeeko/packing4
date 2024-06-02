#pragma once

#include "defs.h"
#include "functional.h"

template<HowToCalGradient how> 
xyt singleGradient(ParticlePair& ijxytt);


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
            xyt f = singleGradient<how>(src[i]);
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
            xyt f = singleGradient<how>(src[i]);
            ptr[ii] -= f;
        }
    }
}

template<HowToCalGradient how> 
GradientBuffer* PairInfo::CalGradient()
{
    static CacheFunction<PairInfo, GradientBuffer> f(calGradient<how>, new GradientBuffer(N));
    return f(this);
}