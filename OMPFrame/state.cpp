#include "pch.h"
#include "state.h"
#include "potential.h"
#include "optimizer.h"

float xyt::amp2(){
	return x * x + y * y + t * t;
}

inline static float randf() {
	return (float)rand() / RAND_MAX;
}


State::State(int N, int sibling)
{
	this->N = N;
	this->id = 1;
	this->sibling_id = sibling;
	configuration = VectorXf::Zero(3 * N);
	gradient = VectorXf::Zero(3 * N);
	boundary = NULL;
}

State::State(VectorXf q, EllipseBoundary* b, int N, int sibling)
{
	this->N = N;
	this->id = 1;
	this->sibling_id = sibling;
	configuration = q;
	gradient = VectorXf::Zero(3 * N);
	boundary = b;
}

void State::randomInitStateCC()
{
	srand(time(0));
	float a = boundary->a - 1;
	float b = boundary->b - 1;
	xyt* q = (xyt*)(void*)configuration.data();
	for (int i = 0; i < N; i++) {
		float r = sqrtf(randf());
		float phi = randf() * (2 * pi);
		q[i].x = a * r * fcos(phi);
		q[i].y = b * r * fsin(phi);
		q[i].t = randf() * pi;
	}
}

float State::initAsDisks(int max_iterations)
{
	// const int max_iterations = 1e5;
	max_gradient_amps.clear();

	for (int i = 0; i < max_iterations; i++) 
	{
		VectorXf g = CalGradient<AsDisks>();
		float gm = maxGradientAbs(g);
		max_gradient_amps.push_back(gm);

		if (gm < 1e-5) {
			return gm;
		}
		else {
			descent(1e-3, g);
		}
	}
	return max_gradient_amps[max_gradient_amps.size() - 1];
}

void State::setBoundary(float a, float b)
{
	id += 1024;
	boundary->setBoundary(a, b);
}

void State::descent(float a, VectorXf& g)
{
	configuration -= a * g;
	id++;
	crashIfDataInvalid();
}

bool isnan(xyt& q) {
	return isnan(q.x) || isnan(q.y) || isnan(q.t);
}

bool isinf(xyt& q) {
	return isinf(q.x) || isinf(q.y) || isinf(q.t);
}


bool outside(xyt& q, float Xmax, float Ymax) {
	return abs(q.x) > Xmax || abs(q.y) > Ymax;
}

string toString(xyt& q) {
	return "(" + to_string(q.x) + ", " + to_string(q.y) + ", " + to_string(q.t) + ")";
}

string toString(EllipseBoundary* eb) {
	return "{ A=" + to_string(eb->a) + ", B=" + to_string(eb->b) + " }";
}

void State::crashIfDataInvalid()
{
#if ENABLE_NAN_CHECK
	xyt* ptr = (xyt*)configuration.data();
	bool flag = true;
	float
		a1 = boundary->a + 1,
		b1 = boundary->b + 1;

	for (int i = 0; i < N; i++) {
		if (isnan(ptr[i]) || isinf(ptr[i])) {
			cout << "Nan data: " << toString(ptr[i]) << endl;
			flag = false;
		}
		if (outside(ptr[i], a1, b1)) {
			cout << "Out of boundary: " << toString(ptr[i]) << "; "
				 << "where: " << toString(boundary) << endl;
			flag = false;
		}
	}
	if (!flag) {
		cout << "In thread " << this->sibling_id << endl;
		throw 114514;
	}
#endif
}

float State::equilibriumGD(int max_iterations)
{
	float step_size = 1e-3;
	float current_min_energy = CalEnergy();

	max_gradient_amps.clear();

	for (int i = 0; i < max_iterations; i++)
	{
		VectorXf g = CalGradient<Normal>();
		float gm = Modify(g);
		max_gradient_amps.push_back(gm);

		// gradient criterion
		if (gm < 1e-2) {	
			break;
		}
		else {
			if ((i + 1) % ENERGY_STRIDE == 0) {
				float E = CalEnergy();
				// energy criterion
				if (E < 5e-5) {
					break;
				}
				// step size (descent speed) criterion
				else if (abs(1 - E / current_min_energy) < 1e-4) {
					step_size /= 2;
					if (step_size < 1e-4) {
						break;
					}
				}
				current_min_energy = std::min(current_min_energy, E);
			}
			descent(step_size, g);
		}
	}
	return CalEnergy();
}

void _gridLocate(State* s, Grid* grid) {
	grid->init(2, s->boundary->a, s->boundary->b);		// including gird->clear()
	grid->gridLocate(s->configuration.data(), s->N);
}

Grid* State::GridLocate()
{
	static CacheFunction<State, Grid> f(_gridLocate, Grid());
	return f(this);
}

PairInfo* State::CollisionDetect()
{
	static CacheFunction<State, PairInfo> f(collisionDetect, PairInfo(N));
	return f(this);
}


/*
	Note: this method has no cache effect
*/
template <HowToCalGradient how>
VectorXf State::CalGradient()
{
	this->CollisionDetect()->CalGradient<how>()->joinTo(this->gradient);
	return this->gradient;
}

float State::CalEnergy()
{
	return this->CollisionDetect()->CalEnergy()->sum();
}
