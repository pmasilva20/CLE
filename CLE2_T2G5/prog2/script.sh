printf "mat128_32 workers 1\n"
for run in {1..10}; do mpiexec -n 2 ./main mat128_32.bin; done
printf "\n mat128_32 workers 2\n"
for run in {1..10}; do mpiexec -n 3 ./main mat128_32.bin; done
printf "\n mat128_32 workers 4\n"
for run in {1..10}; do mpiexec -n 5 ./main mat128_32.bin; done
printf "\n mat128_32 workers 8\n"
for run in {1..10}; do mpiexec -n 9 ./main mat128_32.bin; done

printf "\n\n"

printf "mat128_64 workers 1\n"
for run in {1..10}; do mpiexec -n 2 ./main mat128_64.bin; done
printf "\nmat128_64 workers 2\n"
for run in {1..10}; do mpiexec -n 3 ./main mat128_64.bin; done
printf "\nmat128_64 workers 4\n"
for run in {1..10}; do mpiexec -n 5 ./main mat128_64.bin; done
printf "\nmat128_64 workers 8\n"
for run in {1..10}; do mpiexec -n 9 ./main mat128_64.bin; done

printf "\n\n"

printf "mat128_128 workers 1\n"
for run in {1..10}; do mpiexec -n 2 ./main mat128_128.bin; done
printf "\nmat128_128 workers 2\n"
for run in {1..10}; do mpiexec -n 3 ./main mat128_128.bin; done
printf "\nmat128_128 workers 4\n"
for run in {1..10}; do mpiexec -n 5 ./main mat128_128.bin; done
printf "\nmat128_128 workers 8\n"
for run in {1..10}; do mpiexec -n 9 ./main mat128_128.bin; done

printf "\n\n"

printf "\nmat128_256 workers 1\n"
for run in {1..10}; do mpiexec -n 2 ./main mat128_256.bin; done
printf "\nmat128_256 workers 2\n"
for run in {1..10}; do mpiexec -n 3 ./main mat128_256.bin; done
printf "\nmat128_256 workers 4\n"
for run in {1..10}; do mpiexec -n 5 ./main mat128_256.bin; done
printf "\nmat128_256 workers 8\n"
for run in {1..10}; do mpiexec -n 9 ./main mat128_256.bin; done

printf "\n\n"


printf "\nmat512_32 workers 1\n"
for run in {1..10}; do mpiexec -n 2 ./main mat512_32.bin; done
printf "\nmat512_32 workers 2\n"
for run in {1..10}; do mpiexec -n 3 ./main mat512_32.bin; done
printf "\nmat512_32 workers 4\n"
for run in {1..10}; do mpiexec -n 5 ./main mat512_32.bin; done
printf "\nmat512_32 workers 8\n"
for run in {1..10}; do mpiexec -n 9 ./main mat512_32.bin; done

printf "\n\n"


printf "\nmat512_64 workers 1\n"
for run in {1..10}; do mpiexec -n 2 ./main mat512_64.bin; done
printf "\nmat512_64 workers 2\n"
for run in {1..10}; do mpiexec -n 3 ./main mat512_64.bin; done
printf "\nmat512_64 workers 4\n"
for run in {1..10}; do mpiexec -n 5 ./main mat512_64.bin; done
printf "\nmat512_64 workers 8\n"
for run in {1..10}; do mpiexec -n 9 ./main mat512_64.bin; done

printf "\n\n"


printf "\nmat512_128 workers 1\n"
for run in {1..10}; do mpiexec -n 2 ./main mat512_128.bin; done
printf "\nmat512_128 workers 2\n"
for run in {1..10}; do mpiexec -n 3 ./main mat512_128.bin; done
printf "\nmat512_128 workers 4\n"
for run in {1..10}; do mpiexec -n 5 ./main mat512_128.bin; done
printf "\nmat512_128 workers 8\n"
for run in {1..10}; do mpiexec -n 9 ./main mat512_128.bin; done

printf "\n\n"


printf "\nmat512_256 workers 1\n"
for run in {1..10}; do mpiexec -n 2 ./main mat512_256.bin; done
printf "\nmat512_256 workers 2\n"
for run in {1..10}; do mpiexec -n 3 ./main mat512_256.bin; done
printf "\nmat512_256 workers 4\n"
for run in {1..10}; do mpiexec -n 5 ./main mat512_256.bin; done
printf "\nmat512_256 workers 8\n"
for run in {1..10}; do mpiexec -n 9 ./main mat512_256.bin; done