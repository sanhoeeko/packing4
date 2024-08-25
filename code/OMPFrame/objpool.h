#pragma once

template<typename object, int n>
struct ObjectPool
{
    object* objects[n];
    bool occupied[n];

    ObjectPool() {
        memset(objects, 0, n * sizeof(object*));
        memset(occupied, 0, n * sizeof(bool));
    }

    object*& operator[](int idx) {
        return objects[idx];
    }

    /*
        return -1 if the allocation fails
    */
    int getAvailableId() {
        for (int i = 0; i < n; i++) {
            if (!occupied[i]) {
                occupied[i] = true;
                return i;
            }
        }
        return -1;
    }

    void free(object* obj) {
        for (int i = 0; i < n; i++) {
            if (objects[i] == obj) {
                occupied[i] = false;
            }
        }
    }
};