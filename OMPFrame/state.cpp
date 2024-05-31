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
	return sqrtf(s);
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

VectorXf* State::CalGradientAsDisks() {
	static VectorXf* g = new VectorXf(3 * N);
	CollisionDetect()->CalGradientAsDisks()->joinTo(g);
	OutOfBoundaryPenalty(g);
	return g;
}

void State::initAsDisks()
{
	const int max_iterations = 1e5;

	for (int i = 0; i < max_iterations; i++) 
	{
		VectorXf* g = CalGradientAsDisks();
		float gm = maxGradientAbs(g);
		max_gradient_amps.push_back(gm);

		if (gm < 1e-5) {
			return;
		}
		else {
			descent(1e-2, g);
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
	OutOfBoundaryPenalty(g);
	return g;
}

void State::OutOfBoundaryPenalty(VectorXf* g)
{
	xyt* q = (xyt*)(void*)configuration.data();
	xyt* gxyt = (xyt*)(void*)g->data();

#pragma omp parallel for num_threads(CORES)
	for (int i = 0; i < N; i++) {
		float h = boundary->distOutOfBoundary(q[i]);
		float f = 1.0f * h;
		gxyt[i].x += f * q[i].x;
		gxyt[i].y += f * q[i].y;
	}
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

EllipseBoundary::EllipseBoundary(float a, float b) : a(a), b(b) { 
	a2 = a * a; b2 = b * b;
}

bool EllipseBoundary::maybeCollide(const xyt& q)
{
	static float
		inv_inner_a2 = 1 / (a - 2) * (a - 2),
		inv_inner_b2 = 1 / (b - 2) * (b - 2);
	return
		(q.x) * (q.x) * inv_inner_a2 + (q.y) * (q.y) * inv_inner_b2 > 1;
}

float EllipseBoundary::distOutOfBoundary(const xyt& q)
{
	float f = (q.x) * (q.x) / a2 + (q.y) * (q.y) / b2 - 1;
	if (f < 0)return 0;
	return f + 1e-2f;
}

/*
	require: (x1, y1) in the first quadrant
*/
void EllipseBoundary::solveNearestPointOnEllipse(float x1, float y1, float& x0, float& y0) {
	/*
		Formulae:
		the point (x0, y0) on the ellipse cloest to (x1, y1) in the first quadrant:

			x0 = a2*x1 / (t+a2)
			y0 = b2*y1 / (t+b2)

		where t is the root of

			((a*x1)/(t+a2))^2 + ((b*y1)/(t+b2))^2 - 1 = 0

		in the range of t > -b*b. The initial guess can be t0 = -b*b + b*y1.
	*/
	float t = -b2 + b * y1;
	
	for (int i = 0; i < 4; i++) {
		// Newton root finding. There is always `Ga * Ga + Gb * Gb - 1 > 0`.
		float
			a2pt = a2 + t,
			b2pt = b2 + t,
			ax1 = a * x1,
			by1 = b * y1,
			Ga = ax1 / a2pt,
			Gb = by1 / b2pt,
			G = Ga * Ga + Gb * Gb - 1,
			dG = -2 * ((ax1 * ax1) / (a2pt * a2pt * a2pt) + (by1 * by1) / (b2pt * b2pt * b2pt));
		if (G < 1e-4f) {
			break;
		}
		else {
			t -= G / dG;
		}
	}
	x0 = a2 * x1 / (t + a2);
	y0 = b2 * y1 / (t + b2);
}

Maybe<ParticlePair> EllipseBoundary::collide(int id, const xyt& q)
{
	static float x0, y0, absx0, absy0;

	// q.x,	q.y cannot be both zero because of the `maybeCollide` guard. 
	float absx1 = abs(q.x), absy1 = abs(q.y);
	if (absx1 < 1e-4) {
		x0 = 0; y0 = q.y > 0 ? b : -b;
	}
	else if (absy1 < 1e-4) {
		y0 = 0; x0 = q.x > 0 ? a : -a;
	}
	else {
		solveNearestPointOnEllipse(absx1, absy1, absx0, absy0);
		x0 = q.x > 0 ? absx0 : -absx0;
		y0 = q.y > 0 ? absy0 : -absy0;
	}

	// check if really collide
	float
		dx = q.x - x0, 
		dy = q.y - y0,
		r2 = dx * dx + dy * dy;
	if (r2 >= 1) {
		return Nothing<ParticlePair>();
	}
	
	// calculate the mirror image
	float
		alpha = atan2f(b2 * x0, a2 * y0),
		thetap = pi - q.t + 2 * alpha;
	return Just<ParticlePair>(
		{ id, -1, 2 * dx, 2 * dy, q.t, thetap }
	);
}
