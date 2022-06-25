#include "common.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>


void updateMatrixCPU(double* d_matrices, double* d_results, int n, int m){
  for(int m_id = 0; m_id < m; m_id++){
    int matrixId = m_id * n * n;
    for(int i = 0; i < n-1; i++){

      for(int k = i+1; k < n; k++){
        double pivot = d_matrices[matrixId + i*n + i];
        double term = d_matrices[matrixId + k*n + i] / pivot;

        for(int j=0;j<n;j++){
          d_matrices[matrixId + k*n + j]-= (term * d_matrices[matrixId + i*n + j]);
        }
      }
    }
    d_results[m_id] = 1;
    
    for (int x = 0; x < n; x++){
        d_results[m_id]*= d_matrices[matrixId + x*n + x];
    }
  }
}

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


int main(int argc, char **argv) {
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
    memset(matrices, 0, sizeof(double) * size);

    double *results = (double *) malloc(sizeof(double) * m);
    memset(results, 0, sizeof(double) * m);
    double *matrices_cpu = (double *) malloc(sizeof(double) * size);
    double *results_cpu = (double *) malloc(sizeof(double) * m);

    // read all matrices and store them in memory 
    if (!fread(matrices, 8, size, fp)) {
          exit(1);
    }
    // copy matrices for CPU calculations
    //memcpy(matrices_cpu, matrices, size * 8);

    // create variables to store the matrices and the results for the GPU
    double *d_matrices;
    double *d_results;


    //calculate on the CPU
    calcDeterminantCPUCollumns(matrices, results, n, m);
    
    for (int i = 0; i < m; i++) {
      printf("Matrix cpu %i determinant %+5.3e\n", i, results[i]);
    }
    
    return 0;
}