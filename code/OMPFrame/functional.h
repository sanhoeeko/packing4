#pragma once

#include"defs.h"
#include"io.h"
#include<vector>
#include<functional>
using namespace std;

/*
    Maybe type
*/
template<typename ty>
struct Maybe
{
    bool valid;
    ty obj;

    Maybe() {
        valid = false;
    }
    Maybe(const ty& ref) {
        valid = false;
        obj = ref;
    }
    Maybe(bool validity, const ty& ref) {
        valid = validity;
        obj = ref;
    }
    void clear() {
        if (valid) {
            valid = false;
            obj->clear();
        }
    }
};

template<typename ty>
Maybe<ty> Nothing() {
    return Maybe<ty>();
}

template<typename ty>
Maybe<ty> Just(const ty& x) {
    return Maybe<ty>(true, x);
}

template<typename a, HashFunc hasher>
int anyHasher(const a& x);

/*
    LookupFunc :: 'a[hashable] -> 'b[any]
    Read the result from a database
*/
template<typename a, typename b, int capacity, HashFunc hasher>
struct LookupFunc
{
    b* data;

    LookupFunc() { ; }
    ~LookupFunc() {
        // never deconstruct
    }
    LookupFunc(b func(const a&), vector<a>& inputs) {
        data = (b*)malloc(capacity * sizeof(b));
        int n = inputs.size();
    #pragma omp parallel for num_threads(CORES)
        for (int i = 0; i < n; i++)
        {
            data[i] = func(inputs[i]);
        }
    }
    LookupFunc(std::function<b(const a&)> func, vector<a>& inputs) {
        data = (b*)malloc(capacity * sizeof(b));
        int n = inputs.size();
    #pragma omp parallel for num_threads(CORES)
        for (int i = 0; i < n; i++)
        {
            data[i] = func(inputs[i]);
        }
    }
    b operator()(const a& x) {
        return data[anyHasher<a, hasher>(x)];
    }
};

/*
    D4ScalarFunc :: (float, float, float) -> float
    require: f(x, y, z) = f(-x, -y, z) = f(x, -y, -z) = f(-x, y, -z)
*/
template<int n1, int n2, int n3>
struct D4ScalarFunc
{
    float (*data)[n2][n3];
    size_t capacity = n1 * n2 * n3;

    D4ScalarFunc() {
        data = new float[n1][n2][n3];
    }

    void read(const char* filename) {
        readArrayFromFile<float>((float*)data, capacity, filename);
    }
    void write(const char* filename) {
        writeArrayToFile<float>((float*)data, capacity, filename);
    }
};

/*
    Rolling queue: when roll, delete the first element add add on the last element
*/

template<typename ty, int m>
struct RollList {
    ty data[m];

    ty& operator[](int idx) {
        return data[idx % m];
    }
};