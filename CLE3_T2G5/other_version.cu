/**
 *  \file main.c 
 *
 *  \brief Assignment 3 : Problem 2 - Determinant of a Square Matrix
 *
 *  Processes the Determinant of Square Matrices
 *  The file with the square matrices must be supplied by the user.
 *  CUDA C implementation with both GPU and CPU implementations
 *
 * 
 *  Usage:
 *      \li nvcc main.cu -o prob2
 *      \li ./prob2 <file_name>
 *      \li Example: ./prob2 mat128_32.bin
 * 
 *  \author João Soares (93078) & Pedro Silva (93011)
*/

#include "common.h"
#include <stdio.h>
#include <cuda_runtime.h>
#include "structures.h"

/**
 * \brief Function Calculate Determinant CPU calculates the determinant
 * for each matrix given and stores it's determinant, does it row wise 
 * \param matrices_cpu pointer to matrix array
 * \param determinant_cpu pointer to determinant array
 * \param n matrix order
 * \param m number of matrices
 */
void calcDeterminantCPU(double* matrices_cpu, double* determinant_cpu, int n, int m){
  for(int m_id = 0; m_id < m; m_id++){
    int matrixId = m_id * n * n;
    for(int i = 0; i < n-1; i++){

      for(int k = i+1; k < n; k++){
        double pivot = matrices_cpu[matrixId + i*n + i];
        double term = matrices_cpu[matrixId + k*n + i] / pivot;

        for(int j=0;j<n;j++){
          matrices_cpu[matrixId + k*n + j]-= (term * matrices_cpu[matrixId + i*n + j]);
        }
      }
    }
    determinant_cpu[m_id] = 1;
    
    for (int x = 0; x < n; x++){
        determinant_cpu[m_id]*= matrices_cpu[matrixId + x*n + x];
    }
  }
}
/**
 * \brief Function Calculate Determinant CPU calculates the determinant
 * for each matrix given and stores it's determinant, does it collumn wise 
 * \param matrices_cpu pointer to matrix array
 * \param determinant_cpu pointer to determinant array
 * \param n matrix order
 * \param m number of matrices
 */
void calcDeterminantCPUCollumns(double* d_matrices, double* d_results, int n, int m){
  for(int m_id = 0; m_id < m; m_id++){
    int matrixId = m_id * n * n;
    for(int i = 0; i < n-1; i++){

      for(int k = i+1; k < n; k++){
        double pivot = d_matrices[matrixId + i*n + i];
        double term = d_matrices[matrixId + i*n + k] / pivot;

        for(int j=0;j<n;j++){
          d_matrices[matrixId + j*n + k]-= (term * d_matrices[matrixId + j*n + i]);
        }
      }
    }
    d_results[m_id] = 1;
    
    for (int x = 0; x < n; x++){
        d_results[m_id]*= d_matrices[matrixId + x*n + x];
    }
  }
}



__global__ void calcDeterminant(double* d_matrices, double* d_results) {
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

__global__ void calcDeterminantCollumns(double* d_matrices, double* d_results) {
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

    /* set up the device */
    int dev = 0;

    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("Using Device %d: %s\n", dev, deviceProp.name);
    CHECK(cudaSetDevice(dev));


    /** Initialization FileMatrices*/
    struct FileMatrices file_info;

    //if not enought arguments given
    if (argc != 2)
      { printf ("No file name!\n");
        return EXIT_FAILURE;
      }

    //Code related to reading a file
    if ((file_info.pFile = fopen (argv[1], "r")) == NULL)
    { 
      perror ("error on file opening for reading");
      exit (EXIT_FAILURE);
    }


    
    // Num of matrices
    if (fread(&file_info.numberOfMatrices, sizeof(int), 1, file_info.pFile)==0){
        printf ("Main: Error reading Number of Matrices\n");
    }


    // Order of matrix
    if (fread(&file_info.orderOfMatrices, sizeof(int), 1, file_info.pFile)==0){
        printf("Main: Error reading Order of Matrices\n");
    }
    
    int size = file_info.numberOfMatrices * file_info.numberOfMatrices * file_info.orderOfMatrices;
    
    printf("\nFile %u - Number of Matrices to be read  = %d\n", file_info.id, file_info.numberOfMatrices);

    printf("File %u - Order of Matrices to be read  = %d\n", file_info.id, file_info.orderOfMatrices);


    /** Set File Properties */
    strcpy(file_info.name, argv[1]);

    printf("File name %s\n",file_info.name);


    //Allocate memory for all matrices and results, GPU and CPU
    file_info.matrices = (double *) malloc(sizeof(double) * size);
    file_info.results = (double *) malloc(sizeof(double) * file_info.numberOfMatrices);
    double *matrices_cpu = (double *) malloc(sizeof(double) * size);
    double *results_cpu = (double *) malloc(sizeof(double) * file_info.numberOfMatrices);



    // Read all matrices
    if (fread(file_info.matrices, sizeof(double), size, file_info.pFile) == 0) {
        perror("Main: Error reading Matrix\n");
    }

    // Copy matrices for CPU calculations
    memcpy(matrices_cpu, file_info.matrices, size * 8);

    // create variables to store the matrices and the results for the GPU
    double *d_matrices;
    double *d_results;

    // store the matrices in the GPU
    CHECK(cudaMalloc((void **)&d_matrices, size * sizeof(double)));		// 1d array with all the matrices
    CHECK(cudaMalloc((void **)&d_results, file_info.numberOfMatrices * sizeof(double)));			// 1d array with the results for every matrix


    // fill the matrices
    CHECK(cudaMemcpy(d_matrices, file_info.matrices, size * sizeof(double), cudaMemcpyHostToDevice));


    // add matrix at host side for result checks
    double iStart = seconds();
   

    dim3 grid, block;
    grid.x = file_info.numberOfMatrices;
    block.x = file_info.orderOfMatrices;

    calcDeterminantCollumns<<<grid, block>>>(d_matrices, d_results);
    CHECK(cudaDeviceSynchronize()); 			// wait for kernel to finish
    double iElaps = seconds() - iStart;

    CHECK(cudaMemcpy(file_info.results, d_results, file_info.numberOfMatrices * sizeof(double), cudaMemcpyDeviceToHost));
    CHECK(cudaFree(d_matrices)); 			// clear matrices in the GPU
    CHECK(cudaFree(d_results));        			// clear results in the GPU


    //calculate on the CPU
    calcDeterminantCPU(matrices_cpu, results_cpu, file_info.orderOfMatrices, file_info.numberOfMatrices);
    
    CHECK(cudaDeviceReset());
    for (int i = 0; i < file_info.numberOfMatrices; i++) {
	    printf("Matrix %i determinant %+5.3e\n", i, file_info.results[i]);
      printf("Matrix cpu %i determinant %+5.3e\n", i, results_cpu[i]);
    }
    printf("cenas elapsed %f sec\n", iElaps);
    
    return 0;
}