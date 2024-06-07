#include "pch.h"
#include "optimizer.h"

float maxGradientAbs(VectorXf* g) {
    int n = g->size() / 3;
    xyt* q = (xyt*)(void*)g->data();
    float s = 0;
    for (int i = 0; i < n; i++) {
        float amp2 = q[i].amp2();
        if (s < amp2)s = amp2;
    }
    return sqrtf(s);
}

float sigmaGradientAbs(VectorXf& absg, float mean) {
    VectorXf diff = absg.array() - mean;
    float sq_sum = diff.array().square().sum();
    float std_dev = sqrtf(sq_sum / absg.size());
    return std_dev;
}

void prune(VectorXf* g, float max_element_abs) {
    g->array() = g->array().min(max_element_abs);
    g->array() = g->array().max(-max_element_abs);
}

float Modify(VectorXf* g)
{
    // calculate the mean value and the standard deviation
    VectorXf absg = g->cwiseAbs();
    float 
        mu = absg.mean(),
        sigma = sigmaGradientAbs(absg, mu),
        max_element_abs = mu + 3 * sigma;

    // prune out bad cases
    prune(g, max_element_abs);

    // calculate the max amplitude before normalization
    float res = maxGradientAbs(g);

    // normalize the gradient
    float norm_g = g->norm();
    (*g) /= norm_g;
    return res;
}
