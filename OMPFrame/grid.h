#pragma once

#include "defs.h"
#include "vectorlist.h"
#include "gradient.h"
#include<vector>

using namespace std;
struct State;
struct EllipseBoundary;

const int grid_single_capacity = 16;

void collisionDetect(State* s, PairInfo* pinfo);

struct Grid {
    VectorList<int, CORES, grid_single_capacity>* p;
	float a;									// the "lattice constant" of the grid
	int m, n;									// number of efficient cells of a half axis
	int xshift, yshift;							// number of cells of a half axis
	int lines, cols;							// number of cells of each line of the grid
	int size;									// lines * cols
	int id;										// derived
	int sibling_id;								// dirived
	int* collision_detect_region;				// only 5 cells (including self) are taken into account 
												// in the collision detection of a cell.

	Grid();
	Grid(const Grid& obj);
	void init(float cell_size, float boundary_a, float boundary_b);
	void gridLocate(float* px, int N);

	int xlocate(float x);
	int ylocate(float y);
	VectorList<int, CORES, grid_single_capacity>* loc(int i, int j);
	void add(int thread_idx, int i, int j, int particle_id);
	void toVector();
	void clear();

	void collisionDetectPP(float* x, PairInfo* dst);
	void boundaryCollisionDetectPW(float* x, int N, PairInfo* dst, EllipseBoundary* b);
};