#include"pch.h"
#include"grid.h"

#include"state.h"
#include <cassert>

const int max_particles_in_cell = 32;
const int max_grid_size = 256 * 256;

Grid::Grid() {
	p = new VectorList<int, CORES, grid_single_capacity>[max_grid_size];
	id = -1;
}
Grid::Grid(const Grid& obj)
{
	// memory alloc
	p = new VectorList<int, CORES, grid_single_capacity>[max_grid_size];
	id = obj.id;
}
void Grid::init(float cell_size, float boundary_a, float boundary_b) {
	/*
		Input:
		A: semi major axis
		B: semi minor axis
	*/
	a = cell_size;
	m = (int)ceil(boundary_a / a);
	n = (int)ceil(boundary_b / a);
	xshift = m + 1;
	yshift = n + 1;
	lines = 2 * yshift;
	cols = 2 * xshift;
	size = lines * cols;
	assert(size < max_grid_size);
	collision_detect_region = new int[4] {1 - cols, 1, 1 + cols, cols };
	clear();
}
int Grid::xlocate(float x) {
	return (int)floor(x / a) + xshift;
}
int Grid::ylocate(float y) {
	return (int)floor(y / a) + yshift;
}
VectorList<int, CORES, grid_single_capacity>* Grid::loc(int i, int j) {
	// x -->i, y -->j
	return this->p + j * cols + i;
}
void Grid::add(int thread_idx, int i, int j, int particle_id) {
	(this->p + j * cols + i)->sub_push_back(thread_idx, particle_id);
}
void Grid::toVector()
{
	if constexpr (CORES > 1)		// make 1 core per task zero cost
	{
#pragma omp parallel for num_threads(CORES)
		for (int i = 0; i < size; i++) {
			p[i].toVector();
		}
	}
}
void Grid::clear() {
	for (int i = 0; i < size; i++) {
		p[i].clear();
	}
}
void Grid::gridLocate(float* x, int N) {
	xyt* particles = (xyt*)(void*)x;
	int stride = N / CORES;
#pragma omp parallel num_threads(CORES)
	{
		int idx = omp_get_thread_num();
		int end = idx + 1 == CORES ? N : (idx + 1) * stride;
		for (int cnt = idx * stride; cnt < end; cnt++) {
			int i = xlocate(particles[cnt].x);
			int j = ylocate(particles[cnt].y);
			add(idx, i, j, cnt);
		}
	}
	this->toVector();
}

void Grid::collisionDetectPP(float* x, PairInfo* dst) {
	/*
		Given that a particle is in a certain grid,
		it is only possible to collide with particles in that grid or surrounding grids.
	*/
	xyt* particles = (xyt*)(void*)x;

#pragma omp parallel num_threads(CORES)
	{
		int idx = omp_get_thread_num();
		int n_tasks = cols - 2;
		int start = 1 + idx * (n_tasks / CORES);
		int end = 1 + (idx + 1) * (n_tasks / CORES);
		if (idx + 1 == CORES) end = cols - 1;

		for (int i = start; i < end; i++)  // begin at the (1,1) cell.
		{
			for (int j = 1; j < lines - 1; j++)
			{
				auto* lst = loc(i, j);
				// if there is a particle in the grid:
				if (!lst->empty())
				{
					// for "half-each" surrounding grid (not 9, but 5 cells, including itself): 4 cells different from self
					for (int k = 0; k < 4; k++)
					{
						auto* nlst = lst + collision_detect_region[k];
						if (!nlst->empty())
						{
							// for each particle pair in these two grid:
							for (int ii = 0; ii < lst->size(); ii++) {
								for (int jj = 0; jj < nlst->size(); jj++) {
									int
										p = lst->data[ii], q = nlst->data[jj];
									xyt*
										P = particles + p, * Q = particles + q;
									float
										dx = P->x - Q->x,
										dy = P->y - Q->y,
										r2 = dx * dx + dy * dy;
									if (r2 < 4) {
										dst->info_pp[idx].push_back({ p,q,dx,dy,P->t,Q->t });
									}
								}
							}
						}
					}
					// When and only when collide in one cell, triangular loop must be taken,
					// which ensure that no collision is calculated twice.
					int _n = lst->size();
					for (int ii = 0; ii < _n; ii++) {
						for (int jj = ii + 1; jj < _n; jj++) {
							int
								p = lst->data[ii], q = lst->data[jj];
							xyt*
								P = particles + p, * Q = particles + q;
							float
								dx = P->x - Q->x,
								dy = P->y - Q->y,
								r2 = dx * dx + dy * dy;
							if (r2 < 4) {
								dst->info_pp[idx].push_back({ p,q,dx,dy,P->t,Q->t });
							}
						}
					}
				}
			}
		}
	}
}

void Grid::boundaryCollisionDetectPW(float* x, int N, PairInfo* dst, EllipseBoundary* b) {
	/*
		Note: the actual work of boundary collision detection is delegated to the boundary object.
	*/
	xyt* particles = (xyt*)(void*)x;

#pragma omp parallel num_threads(CORES)
	{
		int idx = omp_get_thread_num();
	#pragma omp for
		for (int i = 0; i < N; i++) {
			if (b->maybeCollide(particles[i])) {
				Maybe<ParticlePair> pair = b->collide(i, particles[i]);
				if (pair.valid) {
					dst->info_pw[idx].push_back(pair.obj);
				}
			}
		}
	}
}

void _collisionDetect(State* s, PairInfo* pinfo) {
	pinfo->clear();
	Grid* grid = s->GridLocate();
	grid->collisionDetectPP(s->configuration.data(), pinfo);
	grid->boundaryCollisionDetectPW(s->configuration.data(), s->N, pinfo, s->boundary);
}
