echo "mat128_32 workers 1"
for run in {1..10}; do mpiexec -n 2 ./main mat128_32.bin; done
echo "mat128_32 workers 2"
for run in {1..10}; do mpiexec -n 3 ./main mat128_32.bin; done
echo "mat128_32 workers 4"
for run in {1..10}; do mpiexec -n 5 ./main mat128_32.bin; done
echo "mat128_32 workers 8"
for run in {1..10}; do mpiexec -n 9 ./main mat128_32.bin; done


echo "mat128_64 workers 1"
for run in {1..10}; do mpiexec -n 2 ./main mat128_64.bin; done
echo "mat128_64 workers 2"
for run in {1..10}; do mpiexec -n 3 ./main mat128_64.bin; done
echo "mat128_64 workers 4"
for run in {1..10}; do mpiexec -n 5 ./main mat128_64.bin; done
echo "mat128_64 workers 8"
for run in {1..10}; do mpiexec -n 9 ./main mat128_64.bin; done

echo "mat128_128 workers 1"
for run in {1..10}; do mpiexec -n 2 ./main mat128_128.bin; done
echo "mat128_128 workers 2"
for run in {1..10}; do mpiexec -n 3 ./main mat128_128.bin; done
echo "mat128_128 workers 4"
for run in {1..10}; do mpiexec -n 5 ./main mat128_128.bin; done
echo "mat128_128 workers 8"
for run in {1..10}; do mpiexec -n 9 ./main mat128_128.bin; done

echo "mat128_256 workers 1"
for run in {1..10}; do mpiexec -n 2 ./main mat128_256.bin; done
echo "mat128_256 workers 2"
for run in {1..10}; do mpiexec -n 3 ./main mat128_256.bin; done
echo "mat128_256 workers 4"
for run in {1..10}; do mpiexec -n 5 ./main mat128_256.bin; done
echo "mat128_256 workers 8"
for run in {1..10}; do mpiexec -n 9 ./main mat128_256.bin; done


echo "mat128_32 workers 1"
for run in {1..10}; do mpiexec -n 2 ./main mat128_32.bin; done
echo "mat128_32 workers 2"
for run in {1..10}; do mpiexec -n 3 ./main mat128_32.bin; done
echo "mat128_32 workers 4"
for run in {1..10}; do mpiexec -n 5 ./main mat128_32.bin; done
echo "mat128_32 workers 8"
for run in {1..10}; do mpiexec -n 9 ./main mat128_32.bin; done


echo "mat512_64 workers 1"
for run in {1..10}; do mpiexec -n 2 ./main mat512_64.bin; done
echo "mat512_64 workers 2"
for run in {1..10}; do mpiexec -n 3 ./main mat512_64.bin; done
echo "mat512_64 workers 4"
for run in {1..10}; do mpiexec -n 5 ./main mat512_64.bin; done
echo "mat512_64 workers 8"
for run in {1..10}; do mpiexec -n 9 ./main mat512_64.bin; done

echo "mat512_128 workers 1"
for run in {1..10}; do mpiexec -n 2 ./main mat512_128.bin; done
echo "mat512_128 workers 2"
for run in {1..10}; do mpiexec -n 3 ./main mat512_128.bin; done
echo "mat512_128 workers 4"
for run in {1..10}; do mpiexec -n 5 ./main mat512_128.bin; done
echo "mat512_128 workers 8"
for run in {1..10}; do mpiexec -n 9 ./main mat512_128.bin; done

echo "mat512_256 workers 1"
for run in {1..10}; do mpiexec -n 2 ./main mat512_256.bin; done
echo "mat512_256 workers 2"
for run in {1..10}; do mpiexec -n 3 ./main mat512_256.bin; done
echo "mat512_256 workers 4"
for run in {1..10}; do mpiexec -n 5 ./main mat512_256.bin; done
echo "mat512_256 workers 8"
for run in {1..10}; do mpiexec -n 9 ./main mat512_256.bin; done