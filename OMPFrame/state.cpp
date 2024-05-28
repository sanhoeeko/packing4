#include"pch.h"
#include "state.h"

State::State(int N)
{
	this->N = N;
	configuration = VectorXf(N);
}

State::State(VectorXf q, EllipseBoundary* b, int N)
{
	this->N = N;
	configuration = q;
	boundary = b;
}

void _calGradient(State* s, VectorXf* g) {
	s->CollisionDetect()->CalGradient()->joinTo(g);
}

VectorXf* State::CalGradient()
{
	static CacheFunction<State, VectorXf> f(_calGradient);
	return f(this);
}

State State::GradientDescent(float a)
{
	VectorXf* g = CalGradient();
	VectorXf new_configuration = this->configuration - a * *g;
	return State(new_configuration, this->boundary, this->N);
}

void _gridLocate(State* s, Grid* grid) {
	grid = new Grid();
	grid->init(2, s->boundary->a, s->boundary->b);
	grid->gridLocate(s->configuration.data(), s->N);
}

Grid* State::GridLocate()
{
	static CacheFunction<State, Grid> f(_gridLocate);
	return f(this);
}

PairInfo* State::CollisionDetect()
{
	static CacheFunction<State, PairInfo> f(collisionDetect);
	return f(this);
}

inline float randf() {
	return (float)rand() / RAND_MAX;
}

State randomInitCC(int N, EllipseBoundary* boundary)
{
	srand(time(0));
	State s(N);
	s.boundary = boundary;
	float* x = s.configuration.data();
	float* y = x + N;
	for (int i = 0; i < N; i++) {
		float r = sqrt(randf());
		float phi = randf() * (2 * pi);
		x[i] = boundary->a * r * cos(phi);
		y[i] = boundary->b * r * sin(phi);
	}
	return s;
}

bool EllipseBoundary::maybeCollide(const xyt& particle)
{
	return false;
}

Maybe<ParticlePair> EllipseBoundary::collide(const xyt& particle)
{
	return Maybe<ParticlePair>();
}
