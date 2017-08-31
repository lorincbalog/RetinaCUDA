# RetinaCUDA
The repository contains a CUDA implementation of the Retinal-Cortical transformation.
The CUDA implementation has been built into shared libraries, dll for Windows and so for Linux users.
Python wrappers are provided to call the library functions.
This project is based on Piotr Ozimek's work at the School of Computing Science at the University of Glasgow.

# Repo structure
bin contains the precompiled shared objects for Linux and the dll for Windows in the Linux and Windows folder, respectively<br />
cpp contains the CUDA code and the C wrappers<br />
py contains the Python wrappers and the Piotr's code used as reference in the Piotr_Ozimek_retina folder<br />
Retinas contains the locations and coefficients representing the retina as samplingfields<br />
Tests contains two tests used to validate the results and measure the performance of the GPU implementation<br />

# Dependencies
Although the binaries should not depend on any libraries which have not been included,
to use the Python wrappers and especially to run the tests the following software dependencies apply:
python 2.7<br />
opencv-python (2.4.13+) -> must be opencv version 2<br />
numpy (1.11.1+)<br />
scipy (0.18.1+)<br />
NOTE: Tests are using camera stream as an input<br />

# Tests
Python tests are located in the Tests folder.
func_test.py contains a correctness test to prove the results are identical with Piotr's results<br />
perf_test.py contains tests to measure the performance of the CUDA implementation,
as well as to compare it with the former Python implementation

# Recompile the binaries
Nvidia CUDA toolkit must be installed and a C++11 compatible compiler is needed.
Use NSight or Visual Studio to build the code under ./cpp into a so or dll.
