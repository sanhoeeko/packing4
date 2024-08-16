#include "pch.h"
#include "state.h"
#include "optimizer.h"
#include "potential.h"
#include "grid.h"
#include "gradient.h"
#include <random>

float xyt::amp2(){
	return x * x + y * y + t * t;
}

State::State(int N)
{
	this->N = N;
	this->sibling_id = 255;  // default: only for test functions
	boundary = NULL;
	configuration = VectorXf::Zero(dof * N);
	grid = Maybe<Grid*>(new Grid());
	pair_info = Maybe<PairInfo*>(new PairInfo(N));
}

State::State(int N, int sibling) : State(N)
{
	this->sibling_id = sibling;
}

State::State(int N, int sibling, EllipseBoundary* b) : State(N, sibling)
{
	boundary = b;
}

State::State(int N, int sibling, EllipseBoundary* b, VectorXf q) : State(N, sibling, b)
{
	configuration = q;
}

/*
	clear cache when the boundary or the configuration is changed
*/
void State::clearCache()
{
	grid.clear();
	pair_info.clear();
}

void State::randomInitStateCC()
{
	// use dist(gen) to generate a random float in [0,1]
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<float> dist(0.0, 1.0);  

	float a = boundary->a - 1;
	float b = boundary->b - 1;
	xyt* q = (xyt*)(void*)configuration.data();
	for (int i = 0; i < N; i++) {
		float r = sqrtf(dist(gen));
		float phi = dist(gen) * (2 * pi);
		q[i].x = a * r * fcos(phi);
		q[i].y = b * r * fsin(phi);
		q[i].t = dist(gen) * pi;
	}
}

float State::initAsDisks(int max_iterations)
{
	/*
		use `ge` as max gradient amplitudes
	*/
	// const int max_iterations = 1e5;
	ge.clear();
	float gm = 0;

	for (int i = 0; i < max_iterations; i++) 
	{
		VectorXf g = CalGradient<AsDisks>();
		gm = maxGradientAbs(g);
		ge.push_back(gm);

		if (gm < 1e-5) {
			return gm;
		}
		else {
			descent(1e-3, g);
		}
	}
	return gm;
}

void State::setBoundary(float a, float b)
{
	boundary->setBoundary(a, b);
	clearCache();
}

void State::descent(float a, VectorXf& g)
{
	configuration -= a * g;
	clearCache();
	crashIfDataInvalid();
}

void State::loadFromData(float* data_src)
{
	/*
		The format of data underlying `data_src` must be `4 * N`
	*/
	memcpy(configuration.data(), data_src, dof * N * sizeof(float));
	clearCache();
}

bool outside(xyt& q, float Xmax, float Ymax) {
	return abs(q.x) > Xmax || abs(q.y) > Ymax;
}

string toString(EllipseBoundary* eb) {
	return "{ A=" + to_string(eb->a) + ", B=" + to_string(eb->b) + " }";
}

void State::crashIfDataInvalid()
{
#if ENABLE_NAN_CHECK || ENABLE_OUT_CHECK
	xyt* ptr = (xyt*)configuration.data();
	bool flag = true;
	float
		a1 = boundary->a + 1,
		b1 = boundary->b + 1;

#pragma omp parallel for
	for (int i = 0; i < N; i++) {
#if ENABLE_NAN_CHECK
		if (isnan(ptr[i]) || isinf(ptr[i])) {
			cout << "Nan data: " << toString(ptr[i]) << endl;
			flag = false;
		}
#endif
#if ENABLE_OUT_CHECK
		if (outside(ptr[i], a1, b1)) {
			cout << "Out of boundary: " << toString(ptr[i]) << "; "
				 << "where: " << toString(boundary) << endl;
			flag = false;
		}
#endif
	}
	if (!flag) {
		cout << "In thread " << this->sibling_id << endl;
		throw 114514;
	}
#endif
}

float State::eqMix(int max_iterations)
{
	ge.clear();
	this->template equilibrium<false, true>(max_iterations, 0.4);
	return this->template equilibrium<false, false>(max_iterations, 1e-3);
}

float State::equilibriumGD(int max_iterations)
{
	ge.clear();
	return this->template equilibrium<false, false>(max_iterations, 1e-4);
}

float State::eqLineGD(int max_iterations)
{
	ge.clear();
	return this->template equilibrium<true, false>(max_iterations, 1e-3);
}

float State::eqLBFGS(int max_iterations)
{
	ge.clear();
	return this->template equilibrium<true, true>(max_iterations, 1e-2);
}

void _gridLocate(State* s, Grid* grid) {
	grid->init(2, s->boundary->a, s->boundary->b);		// including gird->clear()
	grid->gridLocate(s->configuration.data(), s->N);
}

Grid* State::GridLocate()
{
	if (!grid.valid) {
		grid.valid = true;
		_gridLocate(this, grid.obj);
	}
	return grid.obj;
}

PairInfo* State::CollisionDetect()
{
	if (!pair_info.valid) {
		pair_info.valid = true;
		_collisionDetect(this, pair_info.obj);
	}
	return pair_info.obj;
}

float State::CalEnergy()
{
	return this->CollisionDetect()->CalEnergy()->sum();
}

VectorXf State::LbfgsDirection(int iterations)
{
	const int m = 4;					// determine the precision of the inverse Hessian
	const float step_size = 1e-3;
	L_bfgs<m> lbfgs(this);
	StateLoader sl(this); sl.clear();

	VectorXf d(N);
	for (int i = 0; i < iterations; i++) {
		d = lbfgs.CalDirection(sl.s_temp, i);
		lbfgs.update(sl.setDescent(step_size, d), i);
	}
	d = lbfgs.CalDirection(sl.s_temp, iterations);
	return d;
}
