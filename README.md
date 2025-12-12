# Parallel Pattern Recognition (SAD) with Python 

This project implements a high-performance **Parallel Pattern Recognition** algorithm using the Sum of Absolute Differences (SAD) metric. 

## 📋 Prerequisites

To run this project, you need:

1.  **Python 3.8 or higher** (Tested on Python 3.14).


## Required dependencies

This project relies on the following Python libraries:

* **NumPy:** For high-performance array manipulations and AVX2 vectorization.
* **Joblib:** For process-based parallel execution to bypass the GIL.

## ⚙️ How to Run

Simply execute the main script from your terminal.
Change of parameters:
From line 8 to line 11 basic parameters for testing can be changed(long_set length,pattern length,repetitions for testing,number of cores)
In line 18 ,the patterns position(where it is introduced) in the long set can modified.

