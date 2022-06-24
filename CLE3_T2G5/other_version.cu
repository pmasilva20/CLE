#include "common.h"
#include <stdio.h>
#include <cuda_runtime.h>

// https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#block-wide-synchronization
__global__ void updateMatrix(double* d_matrices, double* d_results) {
	int n = blockDim.x;
	for (int iter = 0; iter < n; iter++) {
   
    if (threadIdx.x < iter) 
      continue;

    int matrixId = blockIdx.x * n * n;
    int row = matrixId + threadIdx.x * n;			// current row offset of this (block thread)	
    int iterRow = matrixId + iter * n;

    if (threadIdx.x == iter) {
      if (iter == 0)
        d_results[blockIdx.x] = 1;
      d_results[blockIdx.x] *= d_matrices[iterRow + iter];
      continue;
    }

    double pivot = d_matrices[iterRow + iter];

    double value = d_matrices[row + iter] / pivot;
    for (int i = iter + 1; i < n; i++) {
      d_matrices[row + i] -= d_matrices[iterRow + i] * value; 
    }
    __syncthreads();
  }
}

__global__ void updateMatrixCols(double* d_matrices, double* d_results) {
	int n = blockDim.x;
	for (int iter = 0; iter < n; iter++) {
   
    if (threadIdx.x < iter) 
      continue;

    int matrixId = blockIdx.x * n * n;
    int col = matrixId + threadIdx.x;			// current col offset of this (block thread)	
    int iterCol = matrixId + iter;

    if (threadIdx.x == iter) {
      if (iter == 0)
        d_results[blockIdx.x] = 1;
      d_results[blockIdx.x] *= d_matrices[iterCol + iter * n];
      continue;
    }

    double pivot = d_matrices[iterCol + iter * n];

    double value = d_matrices[col + iter * n] / pivot;
    for (int i = iter + 1; i < n; i++) {
      d_matrices[col + i * n] -= d_matrices[iterCol + i * n] * value; 
    }
    __syncthreads();
  }
}


int main(int argc, char **argv) {
    int dev = 0;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("Using Device %d: %s\n", dev, deviceProp.name);
    CHECK(cudaSetDevice(dev));


    FILE *fp = fopen("./mat128_256.bin", "rb");
    
    // number of matrices
    int m;
    if (!fread(&m, 4, 1, fp)) {
      exit(1);
    }

    // order of the matrix
    int n;
    if (!fread(&n, 4, 1, fp)) {
      exit(1);
    }
    
    int size = n * n * m;
    // stores all the matrices of the file in memory
    double *matrices = (double *) malloc(sizeof(double) * size);
    double *results = (double *) malloc(sizeof(double) * m);

    // read all matrices and store them in memory 
    if (!fread(matrices, 8, size, fp)) {
          exit(1);
    }

    // create variables to store the matrices and the results for the GPU
    double *d_matrices;
    double *d_results;

    // store the matrices in the GPU
    CHECK(cudaMalloc((void **)&d_matrices, size * sizeof(double)));		// 1d array with all the matrices
    CHECK(cudaMalloc((void **)&d_results, m * sizeof(double)));			// 1d array with the results for every matrix


    // fill the matrices
    CHECK(cudaMemcpy(d_matrices, matrices, size * sizeof(double), cudaMemcpyHostToDevice));


    // add matrix at host side for result checks
    double iStart = seconds();
   

    dim3 grid, block;
    grid.x = m;
    block.x = n;

    updateMatrixCols<<<grid, block>>>(d_matrices, d_results);
    CHECK(cudaDeviceSynchronize()); 			// wait for kernel to finish
    double iElaps = seconds() - iStart;

    CHECK(cudaMemcpy(results, d_results, m * sizeof(double), cudaMemcpyDeviceToHost));
    CHECK(cudaFree(d_matrices)); 			// clear matrices in the GPU
    CHECK(cudaFree(d_results));        			// clear results in the GPU
    
    CHECK(cudaDeviceReset());
    for (int i = 0; i < m; i++) {
	    printf("Matrix %i determinant %+5.3e\n", i, results[i]);
    }
    printf("cenas elapsed %f sec\n", iElaps);
    
    return 0;
}