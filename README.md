# SDPL-SLAM
Dynamic SLAM system, that tracks points and lines on both static background and dynamic objects, estimating jointly the camera positions, the map and the objects' motion.


# 2. Prerequisites
We have tested the library in **Mac OS X 10.14** and **Ubuntu 16.04**, but it should be easy to compile in other platforms. 

## c++11, gcc and clang
We use some functionalities of c++11, and the tested gcc version is 9.2.1 (ubuntu), the tested clang version is 1000.11.45.5 (Mac).

## OpenCV
We use [OpenCV](http://opencv.org) to manipulate images and features. Download and install instructions can be found at: http://opencv.org. **Required at least 3.0. Tested with OpenCV 3.4**.

## Eigen3
Required by g2o (see below). Download and install instructions can be found at: http://eigen.tuxfamily.org. **Required at least 3.1.0**.

## g2o (Included in dependencies folder)
We use modified versions of [g2o](https://github.com/RainerKuemmerle/g2o) library to perform non-linear optimizations. The modified libraries (which are BSD) are included in the *dependencies* folder.

## Use Dockerfile for auto installation
For Ubuntu users, a Dockerfile is added for automatically installing all dependencies for reproducible environment, built and tested with KITTI dataset. (Thanks @satyajitghana for the contributions 👍 )


# 3. Building SDPL-SLAM Library

Clone the repository:
```
https://github.com/argyrissm/SDPL-SLAM.git
```

We provide a script `build.sh` to build the *dependencies* libraries and *SDPL-SLAM*. 
Please make sure you have installed all required dependencies (see section 2). 
**Please also change the library file suffix, i.e., '.dylib' for Mac (default) or '.so' for Ubuntu, in the main CMakeLists.txt.**
Then Execute:
```
cd SDPL-SLAM
chmod +x build.sh
./build.sh
```

This will create 

1. **libObjSLAM.dylib (Mac)** or **libObjSLAM.so (Ubuntu)** at *lib* folder,

2. **libg2o.dylib (Mac)** or **libg2o.so (Ubuntu)** at */dependencies/g2o/lib* folder,

3. and the executable **vdo_slam** in *example* folder.

## How to run
To be added

