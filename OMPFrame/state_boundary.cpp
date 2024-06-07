#include "pch.h"
#include "state.h"

EllipseBoundary::EllipseBoundary(float a, float b)
{
	setBoundary(a, b);
}

void EllipseBoundary::setBoundary(float a, float b)
{
	this->a = a; this->b = b;

	// derived
	a2 = a * a; b2 = b * b;
	inv_inner_a2 = 1 / ((a - 2) * (a - 2));
	inv_inner_b2 = 1 / ((b - 2) * (b - 2));
}

bool EllipseBoundary::maybeCollide(const xyt& q)
{
	return (q.x) * (q.x) * inv_inner_a2 + (q.y) * (q.y) * inv_inner_b2 > 1;
}

float EllipseBoundary::distOutOfBoundary(const xyt& q)
{
	float f = (q.x) * (q.x) / a2 + (q.y) * (q.y) / b2 - 1;
	if (f < 0)return 0;
	return f;
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

	for (int i = 0; i < 16; i++) {
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
		if (G < 1e-3f) {
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
	float x0, y0, absx0, absy0;

	// check if the particle is outside the boundary. if so, return a penalty
	// a penalty is marked by {id2 = -114514, theta1 = h}
	float h = distOutOfBoundary(q);
	if (h > 0) {
		return Just<ParticlePair>({ id, -114514, q.x, q.y, h });
	}

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

	// check if really collide: if not, return nothing
	float
		dx = q.x - x0,
		dy = q.y - y0,
		r2 = dx * dx + dy * dy;
	if (r2 >= 1) {
		return Nothing<ParticlePair>();
	}

	// calculate the mirror image
	float
		alpha = atan2f(a2 * y0, b2 * x0),	// the angle of the tangent line
		beta = q.t,
		thetap = 2 * alpha - beta;
	return Just<ParticlePair>(
		{ id, -1, 2 * dx, 2 * dy, q.t, thetap }
	);
}