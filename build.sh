#!/bin/bash
cd OMPFrame
if [ -f OMPFrame.so ]; then rm OMPFrame.so; fi
g++ *.cpp -std=c++17 -o OMPFrame.so -fopenmp -fPIC -shared -Wall -O3
