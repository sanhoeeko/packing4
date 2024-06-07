#pragma once

#include"defs.h"
#include<vector>
using namespace std;

template<typename dtype>
struct static_array {
    int size;
    dtype* data;

    static_array(int size) {
        this->size = size;
        data = new dtype[size];
    }
    void kill() {
        free(data);
    }
};

vector<float> linspace(float start, float stop, int size);
vector<float> linspace_including_endpoint(float start, float stop, int size);
