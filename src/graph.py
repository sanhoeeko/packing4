from functools import lru_cache

import numpy as np


class Graph:
    def __init__(self, N):
        self.N = N
        self.adjacency = np.full((N, 256), -1, dtype=np.int32)
        self.counter = np.zeros((N,), dtype=np.int32)

    def insert(self, idx, pt):
        self.adjacency[idx, self.counter[idx]] = pt
        self.counter[idx] += 1

    def from_delaunay(self, vertex_neighbor_vertices):
        ind_ptr, data = vertex_neighbor_vertices
        for i in range(self.N):
            neighbors = data[ind_ptr[i]:ind_ptr[i + 1]]
            self.adjacency[i, :len(neighbors)] = neighbors
            self.counter[i] = len(neighbors)
        return self

    @lru_cache(maxsize=None)
    def merge(self, modulus):
        res = [set() for _ in range(modulus)]
        homomorphism_mat = np.where(self.adjacency != -1, self.adjacency % modulus, -1)
        j_max = self.N // modulus
        for i in range(modulus):
            for j in range(j_max):
                line = homomorphism_mat[j * modulus + i]
                res[i] |= set(line[line != -1])
        return MergedGraph(res)


class MergedGraph:
    def __init__(self, lst: list[set]):
        self.lst = lst

    @lru_cache(maxsize=None)
    def neighborNums(self):
        # -1: itself is not a neighbor
        return np.array(list(map(len, self.lst))) - 1
