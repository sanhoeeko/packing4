#include "pch.h"
#include "state.h"
#include "gradient.h"

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

std::pair<float, float> State::orderPhi(float gamma, int p)
{

}

VectorXf State::orderS(float gamma)
{
	Graph<neighbors>* graph = contactGraph(gamma);
	xyt* q = (xyt*)configuration.data();
	VectorXf S = VectorXf::Zero(N);
	for (int i = 0; i < N; i++) {
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
