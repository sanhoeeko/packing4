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

State::State(int N)
{
	this->N = N;
	this->id = 1;		// unsafe
	configuration = VectorXf::Zero(3 * N);
	boundary = NULL;
}

State::State(VectorXf q, EllipseBoundary* b, int N)
{
	this->N = N;
	this->id = 1;		// ???
	configuration = q;
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

void State::initAsDisks()
{
	const int max_iterations = 1e5;
	max_gradient_amps.clear();

	for (int i = 0; i < max_iterations; i++) 
	{
		VectorXf* g = CalGradient<AsDisks>();
		float gm = maxGradientAbs(g);
		max_gradient_amps.push_back(gm);

		if (gm < 1e-5) {
			return;
		}
		else {
			descent(1e-3, g);
		}
	}
}

void State::setBoundary(float a, float b)
{
	id += 1024;
	boundary->setBoundary(a, b);
}

void State::descent(float a, VectorXf* g)
{
	configuration -= a * *g;
	id++;
}

float State::equilibriumGD(int max_iterations)
{
	float step_size = 1e-3;
	float current_min_energy = CalEnergy();

	VectorXf* g;
	max_gradient_amps.clear();

	for (int i = 0; i < max_iterations; i++)
	{
		g = CalGradient<Normal>();
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
	static CacheFunction<State, Grid> f(_gridLocate, new Grid());
	return f(this);
}

PairInfo* State::CollisionDetect()
{
	static CacheFunction<State, PairInfo> f(collisionDetect, new PairInfo(N));
	return f(this);
}


/*
	Note: this method has no cache effect
*/
template <HowToCalGradient how>
VectorXf* State::CalGradient()
{
	static VectorXf* g = new VectorXf(3 * N);
	this->CollisionDetect()->CalGradient<how>()->joinTo(g);
	return g;
}

float State::CalEnergy()
{
	return this->CollisionDetect()->CalEnergy()->sum();
}
