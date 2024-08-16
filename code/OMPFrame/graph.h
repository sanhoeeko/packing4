#pragma once

#include "defs.h"
#include "potential.h"

typedef unsigned short particleId_t;
const int neighbors = 8;

struct SegmentDist {
    Matrix4f mat;
    float l;

    SegmentDist(float length);
    float helper(float abac, float bcba, const Vector2f& ab, const Vector2f& ac);
    float inner(float x, float y, float t1, float t2);
    float operator()(ParticlePair& p);
};

template<int max_neighbors>
struct Graph {
    struct Fix_length_array {
        particleId_t arr[max_neighbors];
        particleId_t& operator[](const int idx) { return arr[idx]; }
    };
    vector<Fix_length_array> data;
    vector<int> z;

    Graph(int n) {
        data.resize(n);
        z.resize(n);
        clear();
    }
    void clear() {
        memset(data.data(), -1, max_neighbors * data.size() * sizeof(particleId_t));  // -1 = (bin) 1111 1111
        memset(z.data(), 0, z.size() * sizeof(int));
    }
    void add(int i, int j) {
        if (z[i] == max_neighbors) {
            cout << "Too many neighbors!" << endl;
            throw 1919810;
        }
        else {
            data[i][z[i]] = j;
            z[i]++;
        }
    }
    void add_pair(int i, int j) {
        add(i, j); add(j, i);
    }
};