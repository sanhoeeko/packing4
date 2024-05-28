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
    bool empty() {
        for (int i = 0; i < nodes; i++) {
            if (ns[i] > 0)return false;
        }
        return true;
    }
};

template<typename ty, int nodes, int max_single_capacity>
struct VectorListIter
{
    VectorList<ty, nodes, max_single_capacity>* src;
    int current_sub_vector = 0;
    int sub_idx = 0;

    VectorListIter(VectorList<ty, nodes, max_single_capacity>& vl) {
        src = &vl;
    }
    VectorListIter(VectorList<ty, nodes, max_single_capacity>& vl, int start) {
        src = &vl;
        while (start > src->ns[current_sub_vector]) {
            start -= src->ns[current_sub_vector];
            current_sub_vector++;
        }
        sub_idx = start;
    }
    void next() {
        sub_idx++;
        if (sub_idx >= src->ns[current_sub_vector]) {
            sub_idx = 0;
            current_sub_vector++;
        }
    }
    ty& val() {
        return src->data[current_sub_vector * max_single_capacity + sub_idx];
    }
    bool goes() {
        return current_sub_vector < nodes;
    }
};