cmake_minimum_required(VERSION 3.22)
project(prog1 C)

set(CMAKE_C_STANDARD 11)


include_directories(.)

add_executable(prog1
        main.c
        func.c
        func.h
        preprocessing.c
        preprocessing.h
        problem1.c
        problem1_functions.c
        problem1_functions.h
        structures.c
        structures.h fifo.c fifo.h)

target_link_libraries(prog1 pthread)