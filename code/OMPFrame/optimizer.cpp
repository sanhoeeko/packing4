#include "pch.h"
#include "optimizer.h"
#include <random>

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

VectorXf normalize(const VectorXf& g) {
    return g / g.norm();
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
                return lst[s->sibling_id];
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
    sample in a Gaussian distribution centered at a half of [max_stepsize]
    Return: an array of length [samples]
*/
static std::pair<VectorXf, VectorXf> landscape(State* s, VectorXf& g, float max_stepsize, int samples)
{
    static StateLoaderManager slm;
    VectorXf xs(samples), ys(samples);
    std::default_random_engine generator;
    std::normal_distribution<float> distribution(max_stepsize / 2, max_stepsize / 2);

    StateLoader* s_temp = slm.loader(s);
    int cnt = 0;
    int max_attemps = 12;
    bool flag = false;

    for (int i = 0; i < max_attemps; i++) {
        try {
            float d;
            while (true) {
                d = distribution(generator);
                if (d > 0 && d < max_stepsize)break;
            }
            State* s_prime = s_temp->setDescent(d, g);
            ys[cnt] = s_prime->CalEnergy();
            // if the calculation of energy is successful
            xs[cnt] = d;
            cnt++;
            if (cnt == samples) {
                flag = true;
                break;
            }
        }
        catch (int exception) {
            ;
        }
    }
    if (!flag) {
        // if there are too many fails
        cout << "Fail to find the best step size." << endl;
        throw 114514;
    }
    return {xs, ys};     // for performance, the first and the last element is not calculated
}

static float minimizeCubic(float a, float b, float c, float sc) {
    float
        p1 = -b / (3 * a),
        p2 = sqrt(b * b - 3 * a * c) / (3 * a),
        root[2] = { p1 - p2, p1 + p2 };
    for (int i = 0; i < 2; i++) {
        if (root[i] > 0 && root[i] < sc && 3 * a * root[i] + b > 0) return root[i];
    }
    throw STEP_SIZE_TOO_SMALL;
}

/*
    Return: step size of the equal energy and its corresponding energy
*/
float ERoot(State* s, VectorXf& g, float expected_stepsize) 
{
    static StateLoaderManager slm;
    StateLoader* sl = slm.loader(s);
    float s1 = expected_stepsize;
    float e_ref = s->CalEnergy();
    float e0 = sl->setDescent(s1, g)->CalEnergy() - e_ref;
    if (e0 < 0) {
        throw STEP_SIZE_TOO_SMALL;
    }
    float e1;
    while (true) {
        s1 /= 2;
        e1 = sl->setDescent(s1, g)->CalEnergy() - e_ref;
        if (e1 < 0) break;
    }
    return s1 * 1.5;
}

float BestStepSize(State* s, VectorXf& g, float max_stepsize) {
    const int n_sample = 12;
    float sc = ERoot(s, g, max_stepsize);
    VectorXf xs, ys;
    std::tie(xs, ys) = landscape(s, g, sc, n_sample);
    VectorXf coeffs = polyFit(xs, ys, 3);
    float bs = minimizeCubic(coeffs[3], coeffs[2], coeffs[1], sc);
    if (bs < 1e-5) return 1e-5;
    if (bs > sc) return sc;
    return bs;
}