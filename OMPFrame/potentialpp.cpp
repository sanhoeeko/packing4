#include"pch.h"
#include"potential.h"


Rod::Rod(int n, float d) {
    a = 1,
    b = 1 / (1 + (n - 1) * d / 2.0f),
    c = a - b;
    this->n = n;
    this->rod_d = d * b;
    a_padded = a + 0.01f;    // zero padding, for memory safety
    b_padded = b + 0.01f;
    this->n_shift = -(n - 1) / 2.0f;
    this->inv_disk_R2 = 1 / (b * b);
    this->fv = new D4ScalarFunc<szx, szy, szt>(HashXyt);
}

float Rod::potentialNoInterpolate(const xyt& _q)
{
    const float
        a1 = szx - 1,
        a2 = szy - 1,
        a3 = szt - 1;
    xyt q = transform(_q);
    int
        i = round(q.x * a1),
        j = round(q.y * a2),
        k = round(q.t * a3);
    return fv->data[i][j][k];
}

float Rod::potential(const xyt& q) {
    /*
        q: real x y theta
    */
    return interpolatePotentialSimplex(transform(q));
}

xyt Rod::gradient(const xyt& q) {
    /*
        q: real x y theta
    */
    xyt g = transform_signed(interpolateGradientSimplex(transform(q)));
    bool 
        sign_x = q.x > 0,
        sign_y = q.y > 0;
    if (sign_x)g.x = -g.x;
    if (sign_y)g.y = -g.y;
    if (!(sign_x ^ sign_y))g.t = -g.t;
    return g;
}