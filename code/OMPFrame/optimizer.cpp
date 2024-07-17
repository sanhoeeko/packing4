#include "pch.h"
#include "optimizer.h"

float maxGradientAbs(VectorXf& g) {
    int n = g.size() / dof;
    xyt* q = (xyt*)(void*)g.data();
    float s = 0;
    for (int i = 0; i < n; i++) {
        float amp2 = q[i].amp2();
        if (s < amp2)s = amp2;
    }
    return sqrtf(s);
}

float Modify(VectorXf& g)
{
    // calculate the max amplitude before normalization
    float res = maxGradientAbs(g);

    // normalize the gradient
    float norm_g = g.norm();
    if(norm_g > 0) g /= norm_g;
    return res;
}
