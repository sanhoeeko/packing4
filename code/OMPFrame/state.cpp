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

float State::equilibriumGD(int max_iterations)
{
	/*
		use `ge` as max gradient energies
	*/
	float step_size = 1e-3;
	float current_min_energy = CalEnergy();

	ge.clear();

	for (int i = 0; i < max_iterations; i++)
	{
		VectorXf g = CalGradient<Normal>();
		float gm = Modify(g);

		// gradient criterion
		if (gm < 1e-2) {	
			break;
		}
		else {
			if ((i + 1) % ENERGY_STRIDE == 0) {
				float E = CalEnergy();
				ge.push_back(E);
				// energy criterion
				if (E < 5e-5) {
					break;
				}
				// step size (descent speed) criterion
				else if (abs(1 - E / current_min_energy) < 1e-4) {
					step_size *= 0.8;			// log(0.8)(0.1) ~ 10
					if (step_size < 1e-5) {
						break;
					}
				}
				// moving average
				current_min_energy = (current_min_energy + E) / 2;
			}
			descent(step_size, g);
		}
	}
	return CalEnergy();
}

float State::eqLineGD(int max_iterations)
{
	/*
		use `ge` as max gradient energies
	*/
	float current_min_energy = CalEnergy();
	float step_size = 1e-3;

	ge.clear();

	for (int i = 0; i < max_iterations; i++)
	{
		VectorXf g = CalGradient<Normal>();
		float gm = Modify(g);

		// gradient criterion
		if (gm < 1e-2) {
			break;
		}
		else {
			if ((i + 1) % 10 == 0) {
				float E = CalEnergy();
				ge.push_back(E);
				// energy criterion
				if (E < 5e-5) {
					break;
				}
				// step size (descent speed) criterion
				else if (abs(1 - E / current_min_energy) < 1e-4) {
					break;
				}
				// moving average
				current_min_energy = (current_min_energy + E) / 2;
			}
			try {
				step_size = BestStepSize(this, g, 1.0);
			}
			catch(int exception){
				step_size = 1e-3;
			}
			descent(step_size, g);
		}
	}
	return CalEnergy();
}

float State::eqLBFGS(int max_iterations)
{
	const int m = 10;					// determine the precision of the inverse Hessian
	const float step_size = 1e-3;
	L_bfgs<m> lbfgs(this);

	ge.clear();

	for (int i = 0; i < max_iterations; i++) {
		VectorXf d = lbfgs.CalDirection(this, i);
		descent(step_size, d);
		lbfgs.update(this, i);
		if ((i + 1) % ENERGY_STRIDE == 0) {
			float E = CalEnergy();
			ge.push_back(E);
			// energy criterion
			if (E < 5e-5) {
				break;
			}
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

float State::meanDistance()
{
	return this->CollisionDetect()->meanDistance();
}

float State::meanContactZ()
{
	return this->CollisionDetect()->contactNumberZ() / (float)N;
}
