# packing4
  This project is under development.

## Build in Windows
  - Open the `.sln` file in Visual Studio.
  - Don't forget to enable `/openmp` in the Release mode.
  - Build the project. Ensure that the target file is named `./x64/Release/OMPFrame.dll`.
  - Open the root folder in PyCharm.
  - Run $\to$ Edit Configurations $\to$ Turn the item "script" to "module".
  - Run any script in `./testScripts` for test.
  - Run `./MyFrame.py` to perform a numerical experiment.
  - Run `./analysis.py` to visualize the data.

## Build in Linux
  - Run `cd OMPFrame`
  - Run `g++ *.cpp -std=c++17 -o OMPFrame.so -fopenmp -fPIC -shared -Wall -O3`
  - Run `cd ..`. Run `pip install -r requirements.txt` if necessary.
  - (At the root directory of this project,) run `python3 -m testScripts.[...]` for test, where `[...]` is a file name, for example, `interactive_potential`.
  - Run `python3 MyFrame.py`  to perform a numerical experiment.
  - Run `python3 analysis.py` to visualize the data.