cmake_minimum_required(VERSION 3.21)
project(problem1 C)

set(CMAKE_C_STANDARD 99)

add_executable(problem1 problem1.c problem1_functions.h problem1_functions.c preprocessing.c preprocessing.h)
add_executable(test test.c problem1_functions.h problem1_functions.c preprocessing.c preprocessing.h)
