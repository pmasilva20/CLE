cmake_minimum_required(VERSION 3.22)
project(prog1 C)

set(CMAKE_C_STANDARD 11)


include_directories(.)

add_executable(prog1
        main.c
        assign1_functions.c
        assign1_functions.h
        prob1_processing.c
        prob1_processing.h
        probConst.h
        structures.h shared_region.c shared_region.h assign1_worker.c assign1_worker.h)

target_link_libraries(prog1 pthread)