#pragma once

#include "defs.h"
#include "array.h"

inline static XytPair ZeroXytPair() {
    return { 0,0,0,0,0,0 };
}


/*
    The origin definition of the potential
*/
template<PotentialFunc what>
float Rod::StandardPotential(const xyt& q) {
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
            v += scalarPotential<what>(r2);
        }
    }
    return v;
}

template<PotentialFunc what>
XytPair Rod::StandardGradient(float x, float y, float t1, float t2) {
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
                fr = potentialDR<what>(r2);
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

template<PotentialFunc what>
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
            fv->data[HashXyt(q)] = StandardPotential<what>(inverse(q));
        }
    }
    // fv->write("potential.dat");
}