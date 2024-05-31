#pragma once

#include"defs.h"
#include"io.h"
#include<vector>
#include<functional>
using namespace std;

#undef min

/*
    Maybe type
*/
template<typename ty>
struct Maybe
{
    ty obj;
    bool valid;
};

template<typename ty>
Maybe<ty> Nothing() {
    return { ty(), false };
}

template<typename ty>
Maybe<ty> Just(const ty& x) {
    return { x,true };
}

/*
    ReaderFunc :: 'a[hashable] -> 'b[any]
    Read the result from a database
*/
template<typename a, typename b, size_t capacity>
struct ReaderFunc
{
    b* data;
    function<size_t(const a&)> hasher;

    ReaderFunc(size_t hasher(const a&)) {
        // incomplete initialization
        this->hasher = hasher;
        data = new b[capacity];
    }
    ReaderFunc(b func(const a&), size_t hasher(const a&), vector<a>& inputs) {
        // complete initialization: generate data
        this->hasher = hasher;
        data = (b*)malloc(capacity * sizeof(b));
        size_t n = inputs.size();

    #pragma omp parallel for num_threads(CORES)
        for (int i = 0; i < n; i++)
        {
            data[hasher(inputs[i])] = func(inputs[i]);
        }
    }
    ReaderFunc(const char* database, size_t hasher(const a&)) {
        // complete initialization: read data from a file
        this->hasher = hasher;
        data = new b[capacity];
        readArrayFromFile<b>(data, capacity, database);
    }
    void write(const char* database) {
        writeArrayToFile<b>(data, capacity, database);
    }
    b operator()(const a& x) {
        return data[hasher(x)];
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