cmake_minimum_required(VERSION 3.21)
project(prog2 C)

set(CMAKE_C_STANDARD 99)
set(CMAKE_THREAD_PREFER_PTHREAD TRUE)
set(THREADS_PREFER_PTHREAD_FLAG TRUE)

find_package(Threads REQUIRED)
add_executable(prog2 main.c fifo.c )
target_link_libraries(prog2 Threads::Threads)
target_link_libraries(prog2 m)