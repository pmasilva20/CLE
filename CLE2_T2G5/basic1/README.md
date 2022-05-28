# Compile

mpicc -Wall hello.c -o hello

# Examples

mpiexec -n 4 ./hello

## With cmd args
mpiexec -n 8 ./hello one two three