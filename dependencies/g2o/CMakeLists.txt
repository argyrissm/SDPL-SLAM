CMAKE_MINIMUM_REQUIRED(VERSION 2.6)
SET(CMAKE_LEGACY_CYGWIN_WIN32 0)

PROJECT(g2o)

SET(g2o_C_FLAGS)
SET(g2o_CXX_FLAGS)

# default built type
IF(NOT CMAKE_BUILD_TYPE)
  SET(CMAKE_BUILD_TYPE Release)
ENDIF()

MESSAGE(STATUS "BUILD TYPE:" ${CMAKE_BUILD_TYPE})

SET (G2O_LIB_TYPE SHARED)

# There seems to be an issue with MSVC8
# see http://eigen.tuxfamily.org/bz/show_bug.cgi?id=83
if(MSVC90)
  add_definitions(-DEIGEN_DONT_ALIGN_STATICALLY=1)
  message(STATUS "Disabling memory alignment for MSVC8")
endif(MSVC90)

# Set the output directory for the build executables and libraries
IF(WIN32)
  SET(g2o_LIBRARY_OUTPUT_DIRECTORY ${g2o_SOURCE_DIR}/bin CACHE PATH "Target for the libraries")
ELSE(WIN32)
  SET(g2o_LIBRARY_OUTPUT_DIRECTORY ${g2o_SOURCE_DIR}/lib CACHE PATH "Target for the libraries")
ENDIF(WIN32)
SET(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${g2o_LIBRARY_OUTPUT_DIRECTORY})
SET(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${g2o_LIBRARY_OUTPUT_DIRECTORY})
SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${g2o_RUNTIME_OUTPUT_DIRECTORY})

# Set search directory for looking for our custom CMake scripts to
# look for SuiteSparse, QGLViewer, and Eigen3.
LIST(APPEND CMAKE_MODULE_PATH ${g2o_SOURCE_DIR}/cmake_modules)

# Detect OS and define macros appropriately
IF(UNIX)
  ADD_DEFINITIONS(-DUNIX)
  REMOVE_DEFINITIONS(-D_MSC_VER)
  MESSAGE(STATUS "Compiling on Unix")
ENDIF(UNIX)

# Eigen library parallelise itself, though, presumably due to performance issues
# OPENMP is experimental. We experienced some slowdown with it

FIND_PACKAGE(OpenMP)
SET(G2O_USE_OPENMP OFF CACHE BOOL "Build g2o with OpenMP support (EXPERIMENTAL)")
IF(OPENMP_FOUND AND G2O_USE_OPENMP)
  SET (G2O_OPENMP 1)
  SET(g2o_C_FLAGS "${g2o_C_FLAGS} ${OpenMP_C_FLAGS}")
  SET(g2o_CXX_FLAGS "${g2o_CXX_FLAGS} -DEIGEN_DONT_PARALLELIZE ${OpenMP_CXX_FLAGS}")
  MESSAGE(STATUS "Compiling with OpenMP support")
ENDIF(OPENMP_FOUND AND G2O_USE_OPENMP)

# Compiler specific options for gcc
SET(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -march=native")
SET(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -O3 -march=native")

# activate warnings !!!
SET(g2o_C_FLAGS "${g2o_C_FLAGS} -Wall -W")
SET(g2o_CXX_FLAGS "${g2o_CXX_FLAGS} -Wall -W")

# specifying compiler flags
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${g2o_CXX_FLAGS}")
SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${g2o_C_FLAGS}")

# Find Eigen3
SET(EIGEN3_INCLUDE_DIR ${G2O_EIGEN3_INCLUDE})
FIND_PACKAGE(Eigen3 3.1.0 REQUIRED)
IF(EIGEN3_FOUND)
  SET(G2O_EIGEN3_INCLUDE ${EIGEN3_INCLUDE_DIR} CACHE PATH "Directory of Eigen3")
ELSE(EIGEN3_FOUND)
  SET(G2O_EIGEN3_INCLUDE "" CACHE PATH "Directory of Eigen3")
ENDIF(EIGEN3_FOUND)

# Generate config.h
SET(G2O_CXX_COMPILER "${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER}")
configure_file(config.h.in ${g2o_SOURCE_DIR}/config.h)

find_package(CSparse REQUIRED)
include_directories(${CSPARSE_INCLUDE_DIR})

# Set up the top-level include directories
INCLUDE_DIRECTORIES(
${g2o_SOURCE_DIR}/core
${g2o_SOURCE_DIR}/types
${g2o_SOURCE_DIR}/stuff
${g2o_SOURCE_DIR}/solvers
${G2O_EIGEN3_INCLUDE})


# Include the subdirectories
ADD_LIBRARY(g2o ${G2O_LIB_TYPE}

#types
g2o/types/eigen_types_new.h
g2o/types/types_sba.h
g2o/types/types_six_dof_expmap.h
g2o/types/types_sba.cpp
g2o/types/types_six_dof_expmap.cpp
g2o/types/types_seven_dof_expmap.cpp
g2o/types/types_seven_dof_expmap.h
g2o/types/se3quat.h
g2o/types/se3_ops.h
g2o/types/se3_ops.hpp
g2o/types/edge_xyz_prior.cpp
g2o/types/edge_xyz_prior.h
g2o/types/edge_se3_prior.cpp
g2o/types/edge_se3_prior.h
g2o/types/g2o_types_slam3d_api.h
g2o/types/vertex_pointxyz.cpp
g2o/types/vertex_pointxyz.h
g2o/types/vertex_line.cpp
g2o/types/vertex_line.h
g2o/types/vertex_se3.cpp
g2o/types/vertex_se3.h
g2o/types/edge_se3.cpp
g2o/types/edge_se3.h
g2o/types/edge_se3_ortho_line.h
g2o/types/edge_se3_ortho_line.cpp
g2o/types/isometry3d_mappings.cpp
g2o/types/isometry3d_mappings.h
g2o/types/types_dyn_slam3d.cpp
g2o/types/types_dyn_slam3d.h
g2o/types/dquat2mat.cpp
g2o/types/dquat2mat.h
g2o/types/dquat2mat_maxima_generated.cpp
g2o/types/edge_se3_pointxyz.cpp
g2o/types/edge_se3_pointxyz.h
g2o/types/isometry3d_gradients.cpp
g2o/types/isometry3d_gradients.h
g2o/types/parameter_se3_offset.cpp
g2o/types/parameter_se3_offset.h
g2o/types/types_slam3d.cpp
g2o/types/types_slam3d.h
g2o/types/edge_se3_offset.cpp
g2o/types/edge_se3_offset.h


#core
g2o/core/g2o_core_api.h
g2o/core/base_edge.h
g2o/core/base_binary_edge.h
g2o/core/hyper_graph_action.cpp
g2o/core/base_binary_edge.hpp
g2o/core/hyper_graph_action.h
g2o/core/base_multi_edge.h
g2o/core/hyper_graph.cpp
g2o/core/base_multi_edge.hpp
g2o/core/hyper_graph.h
g2o/core/base_unary_edge.h
g2o/core/linear_solver.h
g2o/core/base_unary_edge.hpp
g2o/core/marginal_covariance_cholesky.cpp
g2o/core/base_vertex.h
g2o/core/marginal_covariance_cholesky.h
g2o/core/base_vertex.hpp
g2o/core/matrix_structure.cpp
g2o/core/batch_stats.cpp
g2o/core/matrix_structure.h
g2o/core/batch_stats.h
g2o/core/openmp_mutex.h
g2o/core/block_solver.h
g2o/core/block_solver.hpp
g2o/core/parameter.cpp
g2o/core/parameter.h
g2o/core/cache.cpp
g2o/core/cache.h
g2o/core/optimizable_graph.cpp
g2o/core/optimizable_graph.h
g2o/core/solver.cpp
g2o/core/solver.h
g2o/core/creators.h
g2o/core/optimization_algorithm_factory.cpp
g2o/core/estimate_propagator.cpp
g2o/core/optimization_algorithm_factory.h
g2o/core/estimate_propagator.h
g2o/core/factory.cpp
g2o/core/optimization_algorithm_property.h
g2o/core/factory.h
g2o/core/sparse_block_matrix.h
g2o/core/sparse_optimizer.cpp
g2o/core/sparse_block_matrix.hpp
g2o/core/sparse_optimizer.h
g2o/core/hyper_dijkstra.cpp
g2o/core/hyper_dijkstra.h
g2o/core/parameter_container.cpp
g2o/core/parameter_container.h
g2o/core/optimization_algorithm.cpp
g2o/core/optimization_algorithm.h
g2o/core/optimization_algorithm_with_hessian.cpp
g2o/core/optimization_algorithm_with_hessian.h
g2o/core/optimization_algorithm_levenberg.cpp
g2o/core/optimization_algorithm_levenberg.h
g2o/core/optimization_algorithm_dogleg.cpp
g2o/core/optimization_algorithm_dogleg.h
g2o/core/optimization_algorithm_gauss_newton.cpp
g2o/core/optimization_algorithm_gauss_newton.h
g2o/core/jacobian_workspace.cpp
g2o/core/jacobian_workspace.h
g2o/core/robust_kernel.cpp
g2o/core/robust_kernel.h
g2o/core/robust_kernel_factory.cpp
g2o/core/robust_kernel_factory.h
g2o/core/robust_kernel_impl.cpp
g2o/core/robust_kernel_impl.h
g2o/core/sparse_optimizer_terminate_action.cpp
g2o/core/sparse_optimizer_terminate_action.h

#stuff
g2o/stuff/string_tools.h
g2o/stuff/color_macros.h
g2o/stuff/macros.h
g2o/stuff/timeutil.cpp
g2o/stuff/misc.h
g2o/stuff/timeutil.h
g2o/stuff/os_specific.c
g2o/stuff/os_specific.h
g2o/stuff/string_tools.cpp
g2o/stuff/property.cpp
g2o/stuff/property.h

#solvers
g2o/solvers/g2o_csparse_extension_api.h
g2o/solvers/csparse_extension.cpp
g2o/solvers/csparse_extension.h
g2o/solvers/csparse_helper.cpp
g2o/solvers/csparse_helper.h
g2o/solvers/g2o_csparse_api.h
#g2o/solvers/solver_csparse.cpp
g2o/solvers/linear_solver_csparse.h
)

target_link_libraries(g2o ${CSPARSE_LIBRARY})
