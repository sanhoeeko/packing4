# packing4
  This project is under development.

### Dependence
  Require `Eigen 3.3.9`.
  Thanks `delaunator-cpp`. (It has been integrated in this project)

## Build in Windows
  - Open the `.sln` file in Visual Studio.
  - Manage Nuget packages $\to$ Install Eigen 3.3.9.
  - Don't forget to enable `/openmp` in the Release mode.
  - Build the project. Ensure that the target file is named `./x64/Release/OMPFrame.dll`.
  - Open the root folder in PyCharm.
  - Run $\to$ Edit Configurations $\to$ Switch the item "script" to "module".
  - Run any script in `./testScripts` for test.
  - Run `./deploy.py` to perform a set of numerical experiments.
  - Run `./analysis.py` to visualize data of a single experiment.
  - Run `./viewData.py` to visualize a set of data.

## Build in Linux
  - Clone Eigen from github $\to$ Extract the headers (a folder named `Eigen`) to `/OMPFrame`.
  - Run `bash build.sh` to build the c++ dynamic-link library.
  - Run `pip install -r requirements.txt` to install python packages if necessary.
  - If your python version is lower than 3.8, run `python3 remove_type_hints.py` to avoid bugs.
  - (At the root directory of this project,) run `python3 -m testScripts.[...]` for test, where `[...]` is a file name, for example, `interactive_potential`.
  - Run `python3 deploy.py` to perform a set of numerical experiments.
