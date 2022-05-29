cmake_minimum_required(VERSION 3.22)
project(prog1 C)

set(CMAKE_C_STANDARD 11)

find_package(MPI REQUIRED)
include_directories(/usr/include/mpi/)
include_directories(${MPI_INCLUDE_PATH})

SET(CMAKE_C_COMPILER mpicc)
SET(CMAKE_CXX_COMPILER mpicxx)

add_executable(prog1
        main.c
        prob1_processing.c
        prob1_processing.h
        structures.h)
