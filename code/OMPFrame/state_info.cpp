#include "pch.h"
#include "state.h"
#include "gradient.h"
#include "delaunator.hpp"

void _delaunay(int n_sites, vector<float>& points, Graph<neighbors>* graph) {
	delaunator::Delaunator d(points);
	for (size_t i = 0; i < d.triangles.size(); i += 3) {
		int p1 = d.triangles[i] % n_sites;
		int p2 = d.triangles[i + 1] % n_sites;
		int p3 = d.triangles[i + 2] % n_sites;
		graph->add_pair_if_hasnot(p1, p2);
		graph->add_pair_if_hasnot(p1, p3);
		graph->add_pair_if_hasnot(p2, p3);
	}
}

void delaunayTriangulate(float gamma, State* s, Graph<neighbors>* graph) {
	/*
		We use 3 - point approximation to describe the shape of the rod.
		When the density is higher, more points will be required.
	*/
	struct Point { float x, y; };
	const int n = 3;					// n should be odd
	const int m = (n - 1) / 2;
	float r = 1 - 1 / gamma;
	vector<Point> points; points.resize(n * s->N);
	xyt* ptr = (xyt*)s->configuration.data();
	
	// convert rod-like particles to sites
	for (int i = 0; i < s->N; i++) {
		float
			dx = r * fcos(ptr[i].t) / m,
			dy = r * fsin(ptr[i].t) / m;
		for (int k = -m; k <= m; k++) {
			float
				x = ptr[i].x + k * dx,
				y = ptr[i].y + k * dy;
			points[(m + k) * s->N + i] = { x,y };
		}
	}

	// execute the delaunay triangulation
	vector<float> pts = vector<float>((float*)points.data(), (float*)(points.data() + points.size()));
	_delaunay(s->N, pts, graph);
}

float State::meanDistance(float gamma)
{
	return this->CollisionDetect()->meanDistance(gamma);
}

float State::meanContactZ(float gamma)
{
	return this->CollisionDetect()->contactNumberZ(gamma) / (float)N;
}

Graph<neighbors>* State::contactGraph(float gamma)
{
	return this->CollisionDetect()->toGraph(gamma);
}

Graph<neighbors>* State::voronoiGraph(float gamma)
{
	if (!voronoi.valid) {
		voronoi.valid = true;
		delaunayTriangulate(gamma, this, voronoi.obj);
	}
	return voronoi.obj;
}

VectorXcf State::orderPhi(float gamma, int p)
{
	Graph<neighbors>* graph = voronoiGraph(gamma);
	xyt* q = (xyt*)configuration.data();
	VectorXcf Phi = VectorXcf::Zero(N);
	for (int i = 0; i < N; i++) {
		if (graph->z[i] == 0) {
			Phi[i] = 0; continue;
		}
		complex<float> phi = 0;
		for (int k = 0; k < graph->z[i]; k++) {
			int j = graph->data[i][k];
			float
				x = q[j].x - q[i].x,
				y = q[j].y - q[i].y,
				theta_ij = atan2f(y, x);
			phi += std::exp(std::complex<float>(0, p * theta_ij));
		}
		Phi[i] = phi / (float)graph->z[i];
	}
	return Phi;
}

complex<float> State::orderPhi_ave(float gamma, int p)
{
	return orderPhi(gamma, p).mean();
}

VectorXf State::orderS(float gamma)
{
	Graph<neighbors>* graph = voronoiGraph(gamma);
	xyt* q = (xyt*)configuration.data();
	VectorXf S = VectorXf::Zero(N);
	for (int i = 0; i < N; i++) {
		if (graph->z[i] == 0) {
			S[i] = 0; continue;
		}
		float s = sin(2 * q[i].t);
		float c = cos(2 * q[i].t);
		for (int k = 0; k < graph->z[i]; k++) {
			int j = graph->data[i][k];
			s += sin(2 * q[j].t);
			c += cos(2 * q[j].t);
		}
		float Si = sqrt(s * s + c * c) / (2 * (1 + graph->z[i]));
		S[i] = Si;
	}
	return S;
}

float State::orderS_ave(float gamma)
{
	return orderS(gamma).mean();
}
