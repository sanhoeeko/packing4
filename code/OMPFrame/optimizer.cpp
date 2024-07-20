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

struct StateLoader {
    State* s_ref;
    State* s_temp;

    StateLoader(State* s) {
        s_ref = s;
        s_temp = new State(s->N);
        s_temp->boundary = s->boundary;
    }
    StateLoader* redefine(State* s) {
        s_ref = s;
        s_temp->boundary = s_ref->boundary;
        return this;
    }
    State* clear() {
        s_temp->loadFromData(s_ref->configuration.data());
        return s_temp;
    }
    State* setDescent(float a, VectorXf& g) {
        clear();
        s_temp->descent(a, g);
        return s_temp;
    }
};

/*
    Reuse state loaders to prevent memory leakage.
*/
struct StateLoaderManager
{
    vector<StateLoader*> lst;

    StateLoader* loader(State* s) {
        if (s->sibling_id < lst.size()) {
            if (lst[s->sibling_id] == NULL) {
                lst[s->sibling_id] = new StateLoader(s);
            }
            else {
                return lst[s->sibling_id]->redefine(s);
            }
        }
        else {
            int n = lst.size();
            int m = s->sibling_id + 1;
            lst.resize(m);
            memset(lst.data() + n, 0, (m - n) * sizeof(StateLoader*));
            lst[s->sibling_id] = new StateLoader(s);
            return lst[s->sibling_id];
        }
    }
};

static VectorXf polyFit(VectorXf& x, VectorXf& y, int deg) {
    MatrixXf mtxVandermonde(x.size(), deg + 1); // Vandermonde matrix of X-axis coordinate vector of sample data
    VectorXf vectColVandermonde = x; // Vandermonde column
    VectorXf coeffs;

    mtxVandermonde.col(0) = VectorXf::Constant(x.size(), 1, 1);
    mtxVandermonde.col(1) = vectColVandermonde;

    // construct Vandermonde matrix column by column
    for (int i = 2; i < deg + 1; i++) {
        vectColVandermonde = vectColVandermonde.array() * x.array();
        mtxVandermonde.col(i) = vectColVandermonde;
    }

    // calculate coefficients vector of fitted polynomial
    coeffs = (mtxVandermonde.transpose() * mtxVandermonde).ldlt().solve(mtxVandermonde.transpose() * y);

    return coeffs;
}

/*
    Return: an array of length [samples + 1]
*/
static VectorXf landscape(State* s, VectorXf& g, float max_stepsize, int samples)
{
    static StateLoaderManager slm;
    VectorXf res(samples + 1);
    float d_stepsize = max_stepsize / samples;
    State* s_temp = slm.loader(s)->clear();

    for (int i = 1; i < samples; i++) {
        s_temp->descent(d_stepsize, g);
        res[i] = s_temp->CalEnergy();
    }
    res[0] = 0;     // res[last] = ec
    return res;     // for performance, the first and the last element is not calculated
}

static float minimizeCubic(float a, float b, float c, float sc) {
    float
        p1 = -b / (3 * a),
        p2 = sqrt(b * b - 3 * a * c) / (3 * a),
        root = p1 - p2;
    if (root > 0 && root < sc) {
        return root;
    }
    else {
        return p1 + p2;
    }
}

/*
    Return: step size of the equal energy and its corresponding energy
*/
std::pair<float, float> ERoot(State* s, VectorXf& g, float expected_stepsize) 
{
    static StateLoaderManager slm;
    StateLoader* sl = slm.loader(s);
    float s1 = expected_stepsize;
    float e_ref = s->CalEnergy();
    float e0 = sl->setDescent(s1, g)->CalEnergy() - e_ref;
    if (e0 < 0) {
        return { -1.0f , e0};
    }
    float e1;
    while (true) {
        s1 /= 2;
        e1 = sl->setDescent(s1, g)->CalEnergy() - e_ref;
        if (e1 < 0) break;
    }
    float s0 = s1 * 2;
    float sc = (e1 * s0 - e0 * s1) / (e1 - e0);
    float sc_cache = 0;
    float ec;
    for (int i = 0; i < 8; i++) {   // usually done within 4 iterations
        ec = sl->setDescent(sc, g)->CalEnergy() - e_ref;
        if (abs(ec) < 1e-3 || abs(sc - sc_cache) < 1e-5) break;
        if (ec * e1 > 0) {
            s0 = sc; e0 = ec; 
        }
        else {
            s1 = sc; e1 = ec;
        }
        sc_cache = sc;
        sc = (e1 * s0 - e0 * s1) / (e1 - e0);
    }
    return { sc, ec };
}

float BestStepSize(State* s, VectorXf& g, float max_stepsize) {
    const int n_sample = 10;
    float sc, ec;
    std::tie(sc, ec) = ERoot(s, g, max_stepsize);
    if (sc == -1.0f) {
        return max_stepsize;    // directly descent
    }
    VectorXf ys = landscape(s, g, sc, n_sample); 
    ys[ys.size() - 1] = ec;
    vector<float> _xs = linspace_including_endpoint(0, sc, n_sample + 1);
    VectorXf xs = Map<VectorXf>(_xs.data(), _xs.size());
    VectorXf coeffs = polyFit(xs, ys, 3);
    return minimizeCubic(coeffs[3], coeffs[2], coeffs[1], sc);
}