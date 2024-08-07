#!/bin/bash

# process python scripts
cd code
python3 remove_type_hints.py

# build DLL
cd OMPFrame
if [ -f OMPFrame.so ]; then rm OMPFrame.so; fi
g++ *.cpp -std=c++17 -o OMPFrame.so -fopenmp -fPIC -shared -Wall -O3
