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
};

template<typename ty>
Maybe<ty> Nothing() {
    return { false };
}

template<typename ty>
Maybe<ty> Just(const ty& x) {
    return { true, x };
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

    LookupFunc(b func(const a&), vector<a>& inputs) {
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

template<typename a, typename b>
struct CacheFunction 
{
    b* cache = NULL;
    function<void(a*, b*)> func;    // function in fishing format. a: input type; b: result type

    CacheFunction(void f(a*,b*), b* new_cache) {
        this->func = function<void(a*, b*)>(f);
        this->cache = new_cache;
    }

    b* operator()(a* x) {
        if (x->id == cache->id) {
            return cache;
        }
        else {
            cache->id = x->id;
            func(x, cache);         // cache = f(x)
            return cache;
        }
    }
};