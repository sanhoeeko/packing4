#include "pch.h"
#include "state.h"
#include "potential.h"

float xyt::amp2(){
	return x * x + y * y + t * t;
}

inline static float randf() {
	return (float)rand() / RAND_MAX;
}

inline float maxGradientAbs(VectorXf* g) {
	int n = g->size() / 3;
	xyt* q = (xyt*)(void*)g->data();
	float s = 0;
	for (int i = 0; i < n; i++) {
		float amp2 = q[i].amp2();
		if (s < amp2)s = amp2;
	}
	return s;
}

inline bool isDataValid(VectorXf* q) {

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
	xyt* q = (xyt*)(void*)configuration.data();
	for (int i = 0; i < N; i++) {
		float r = sqrtf(randf());
		float phi = randf() * (2 * pi);
		q[i].x = boundary->a * r * fcos(phi);
		q[i].y = boundary->b * r * fsin(phi);
	}
}

void State::initAsDisks()
{
	const int max_iterations = 1e5;
	static VectorXf* g = new VectorXf(N * 3);
	for (int i = 0; i < max_iterations; i++) {
		CollisionDetect()->CalGradientAsDisks()->joinTo(g);
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

/*
	Note: this method has no cache effect
*/
VectorXf* State::CalGradient()
{
	static VectorXf* g = new VectorXf(3 * N);
	CollisionDetect()->CalGradient()->joinTo(g);
	return g;
}

void State::descent(float a, VectorXf* g)
{
	configuration -= a * *g;
	id++;
}

State State::GradientDescent(float a)
{
	VectorXf* g = CalGradient();
	VectorXf new_configuration = this->configuration - a * *g;
	return State(new_configuration, this->boundary, this->N);
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

EllipseBoundary::EllipseBoundary(float a, float b) : a(a), b(b) { ; }

bool EllipseBoundary::maybeCollide(const xyt& particle)
{
	return false;
}

Maybe<ParticlePair> EllipseBoundary::collide(const xyt& particle)
{
	return Maybe<ParticlePair>();
}
