cmake_minimum_required(VERSION 3.22)
project(prog1 C)

set(CMAKE_C_STANDARD 11)


include_directories(.)

add_executable(prog1
        main.c
        func.c
        func.h
        prob1_processing.c
        prob1_processing.h
        probConst.h
        structures.c
        structures.h fifo.c fifo.h worker.c worker.h)

target_link_libraries(prog1 pthread)