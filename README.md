# Visual Odometry Kit

Some algorithm modules for VO, the Kit just for verification. May be **not enough stable ！**

## Prerequisites

This is tested in **Ubuntu 16.04**.

### C++11

### OpenCV

We use OpenCV 3.2.0.

### Eigen

we use Eigen 3.2.29


## Modules

### 1.math_utils

some Mathematical tools

- Nonlinear least-square optimizer for simple problem. (lack of sparse solve and marginalization etc.)

- base function for SO3 SE3 quaternion max min etc.

### 2.alignment

The module contains feature alignment and sparse image alignment based on **inverse compositional**.

TODO NEXT


## Build and Run

After clone it, build:
``` shell
 mkdir build && cd build
 cmake ..
 make -j
 ```
Then run their test demos

``` shell
 # math_utils
 ./bin/test_optimizer
 ```