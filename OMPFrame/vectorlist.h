#pragma once

#include<vector>
using namespace std;

template<typename ty, int nodes, int max_single_capacity>
struct VectorList
{
    ty* data;
    int ns[nodes];

    VectorList() {
        data = (ty*)malloc(nodes * max_single_capacity * sizeof(ty));
        memset(ns, 0, nodes * sizeof(int));
    }
    void sub_push_back(int i, const ty& obj) {
        data[i * max_single_capacity + ns[i]] = obj;
        ns[i]++;
    }
    void clear() {
        memset(ns, 0, nodes * sizeof(int));
    }
    void toVector() {
        int current_n_sum = ns[0];
        for (int i = 1; i < nodes; i++) {
            memcpy(data + current_n_sum, data + i * max_single_capacity, ns[i] * sizeof(ty));
            current_n_sum += ns[i];
            ns[i] = 0;
        }
        ns[0] = current_n_sum;
    }
    // the following two function are only called after .toVector()
    int size() {
        return ns[0];
    }
    bool empty() {
        return ns[0] == 0;
    }
};