/**
 *  \file mainColumns.cu 
 *
 *  \brief Assignment 3 : Compute the Value of Determinant by Matrix Columns
 *
 *  Processes the Determinant of Square Matrices where the threads in a block thread process successive matrix columns
 *  The file with the square matrices must be supplied by the user.
 *  GPU Programming implementation.
 *
 * 
 *  Usage:
 *      \li make
 *      \li ./mainColumns <file_name>
 *      \li Example: ./mainColumns mat128_32.bin
 * 
 *  \author João Soares (93078) & Pedro Silva (93011)
*/

#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include "common.h"
#include <cuda_runtime.h>

/**
 *   program configuration
 */

#ifndef SECTOR_SIZE
# define SECTOR_SIZE  512
#endif
#ifndef N_SECTORS
# define N_SECTORS    (1 << 21)                            // it can go as high as (1 << 21)
#endif

/* Allusion to internal functions */

__global__ static void computeDeterminantByColumnsOnGPU(double* d_matrices, double* d_results);

void computeDeterminantByColumnsOnCPU(double* matrices_cpu, double* determinant_cpu, int n, int m);

/**
 * \brief Get the Delta time 
 * 
 * \return double 
 */
static double get_delta_time(void);

/**
 * \brief Print Determinant Results 
 * 
 * \param numberOfMatrices Number of Matrices/Results
 * \param resultsDeterminant Array with Determinant Results
 */
void printResults(int numberOfMatrices, double * resultsDeterminant);


/**
 * \brief Compare the Result Determinant pressed obtained from Device and Host
 * 
 * @param resultsDeterminantDevice Determinant Results Device
 * @param resultsDeterminantHost Determinant Results Host
 * @param numberOfMatrices Number of Matrices/Results
 */
void compareResults(double* resultsDeterminantDevice, double* resultsDeterminantHost,int numberOfMatrices);


/**
 * \brief Main Program
 *  
 * Instantiation of the processing configuration.
 * 
 * \param argc number of words of the command line
 * \param argv list of words of the command line
 * \return status of operation 
 */

int main (int argc, char **argv)
{
  printf("%s Starting...\n", argv[0]);

  if (sizeof (unsigned int) != (size_t) 4)
     // It fails with prejudice if an integer does not have 4 bytes
     return 1;                                            

  
  /* Set up the device */
  int dev = 0;
  
  /* File Reader */
  FILE *pFile;
  
  /** Number of Matrice **/
  int numberOfMatrices;
  
  /** Order of Matrices **/
  int orderOfMatrices;


  cudaDeviceProp deviceProp;
  CHECK (cudaGetDeviceProperties (&deviceProp, dev));
  printf("Using Device %d: %s\n", dev, deviceProp.name);
  CHECK (cudaSetDevice (dev));

  /** Try to Read Matrix File **/
  if (argc != 2){ 
    printf ("No file name!\n");
    return EXIT_FAILURE;
  }
 
  if ((pFile = fopen (argv[1], "r")) == NULL){ 
    perror ("error on file opening for reading");
    exit (EXIT_FAILURE);
  }

  /** Obtain Number of Matrices in File **/
  if (fread(&numberOfMatrices, sizeof(int), 1, pFile)==0){
      printf ("Main: Error reading Number of Matrices\n");
  }

  /** Obtain Order of the Matrices in File **/ 
  if (fread(&orderOfMatrices, sizeof(int), 1, pFile)==0){
      printf("Main: Error reading Order of Matrices\n");
  }

  printf("\nFile - Number of Matrices to be read  = %d\n", numberOfMatrices);

  printf("File - Order of Matrices to be read  = %d\n", orderOfMatrices);


  /* Create Memory Spaces in host and device memory where the Matrix and Matrix Results will be stored */

  /** Matrices Host **/
  double *matricesHost = (double *)malloc(sizeof(double) * numberOfMatrices * orderOfMatrices * orderOfMatrices);
  
  /** Determinant Results Host **/
  double *resultsDeterminantHost = (double *)malloc(sizeof(double) * numberOfMatrices);

  /** Matrices Device/GPU **/
  double *matricesDevice;
 
  /** Determinant Results Device/GPU **/
  double *resultsDeterminantDevice;

  /** Allocate Matrices Device & Determinant Results in Global Memory **/
  CHECK(cudaMalloc((void **)&matricesDevice, numberOfMatrices * orderOfMatrices * orderOfMatrices * sizeof(double)));		
  CHECK(cudaMalloc((void **)&resultsDeterminantDevice, numberOfMatrices * sizeof(double)));
  
  //TODO: Depois se necessário alocar order e numero matrices


  /* Initialize the Host Data with Matrices of the file */

  /** Number of Matrices Read */
  int numberMatricesRead=0;
  
  (void) get_delta_time ();

  if (!fread(matricesHost, sizeof(double), numberOfMatrices * orderOfMatrices * orderOfMatrices, pFile))
  {
    printf("Error: Error reading matrices from file\n");
    return EXIT_FAILURE;
  }  
  printf ("The initialization of Host data took %.3e seconds\n", get_delta_time ());

  /* Copy the host data to the device memory */

  (void) get_delta_time ();
  
  CHECK(cudaMemcpy(matricesDevice, matricesHost, numberOfMatrices * orderOfMatrices * orderOfMatrices * sizeof(double), cudaMemcpyHostToDevice));
  
  printf ("The transfer of Matrices from the host to the device took %.3e seconds\n", get_delta_time ());

  /* Run the computational kernel */
  unsigned int gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ;
  int n_sectors, sector_size;

  n_sectors = N_SECTORS;
  sector_size = SECTOR_SIZE;
  blockDimX = orderOfMatrices;                             // optimize!
  blockDimY = 1 << 0;                                      // optimize!
  blockDimZ = 1 << 0;                                      // do not change!
  gridDimX = numberOfMatrices;                             // optimize!
  gridDimY = 1 << 0;                                       // optimize!
  gridDimZ = 1 << 0;                                       // do not change!

  dim3 grid (gridDimX, gridDimY, gridDimZ);
  dim3 block (blockDimX, blockDimY, blockDimZ);

  /**if ((gridDimX * gridDimY * gridDimZ * blockDimX * blockDimY * blockDimZ) != n_sectors)
     { printf ("Wrong configuration!\n");
       return 1;
     }
  **/
  

  /* Compute Determinant of Matrizes on Device */
  (void) get_delta_time ();
  
  computeDeterminantByColumnsOnGPU <<<grid, block>>> (matricesDevice,resultsDeterminantDevice);

  /** Wait for Kernel to Finish **/
  CHECK (cudaDeviceSynchronize ());                            
  
  double deviceTime=get_delta_time();

  /** Check for Kernel Errors */
  CHECK (cudaGetLastError());                                 

  /** Determinant Results from Device/GPU **/
  double *resultsDeterminantFromDevice = (double *)malloc(sizeof(double) * numberOfMatrices);

  /* Copy kernel result back to host side */
  CHECK(cudaMemcpy(resultsDeterminantFromDevice, resultsDeterminantDevice, numberOfMatrices * sizeof(double), cudaMemcpyDeviceToHost));
  printf ("The transfer of Determinant Results from the device to the host took %.3e seconds\n", get_delta_time ());

  /** Free Matrices from Device/GPU  Global Memory **/
  CHECK (cudaFree (matricesDevice));

  /** Free Results Determinant Device/GPU Global Memory **/
  CHECK (cudaFree (resultsDeterminantDevice));

  /* Reset the Device / GPU */
  CHECK (cudaDeviceReset ());

  /* Compute Determinant of Matrizes on CPU */
  (void) get_delta_time ();
  
  computeDeterminantByColumnsOnCPU(matricesHost,resultsDeterminantHost,orderOfMatrices,numberOfMatrices);
  
  double cpuTime=get_delta_time();
 
  printResults(numberOfMatrices,resultsDeterminantFromDevice);
  
  /** Compare Determinant Results from Host and Device **/
  compareResults(resultsDeterminantFromDevice,resultsDeterminantHost,numberOfMatrices);

  printf("\nThe CPU Kernel took %.3e seconds to run\n",cpuTime);
  printf("The Device CUDA Kernel <<<(%d,%d,%d), (%d,%d,%d)>>> took %.3e seconds to run\n",
         gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, deviceTime);
  
  /* Free host memory */
  free (matricesHost);
  free (resultsDeterminantHost);
  free (resultsDeterminantFromDevice);

  return 0; 
}




//TODO: Colocar aqui versão CPU alterar formatar
//TODO alterar função E Documentação
void computeDeterminantByColumnsOnCPU(double* d_matrices, double* d_results, int n, int m){
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




//TODO alterar função E Documentação
__global__ void computeDeterminantByColumnsOnGPU(double* d_matrices, double* d_results) {
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

/**
 * \brief Print Determinant Results 
 * 
 * \param numberOfMatrices Number of Matrices/Results
 * \param resultsDeterminant Array with Determinant Results
 */
void printResults(int numberOfMatrices, double *resultsDeterminant){
  for (int a = 0; a < numberOfMatrices; a++) {
      printf("Matrix %d :\n", a + 1);
      printf("The determinant is %.3e\n", resultsDeterminant[a]);
  }
}

/**
 * \brief Compare the Result Determinant pressed obtained from Device and Host
 * 
 * @param resultsDeterminantDevice Determinant Results Device
 * @param resultsDeterminantHost Determinant Results Host
 * @param numberOfMatrices Number of Matrices/Results
 */
void compareResults(double* resultsDeterminantDevice, double* resultsDeterminantHost,int numberOfMatrices){

  bool difference=false;

  char resultDevice[1024];

  char resultHost[1024];

  for(int i = 0; i < numberOfMatrices; i++){

    snprintf(resultDevice, sizeof(resultDevice), "%.3e",resultsDeterminantDevice[i]);
    snprintf(resultHost, sizeof(resultHost), "%.3e", resultsDeterminantHost[i]);

    if(strcmp(resultDevice,resultHost)!=0){
      printf("\nResults from Device different compared with Results from Host!\n");
      difference=true;
      break;
    }

  }
  if(!difference){
    printf("\nResults from Device equal to Results from Host!\n");
  }

}

/**
 * \brief Get the Delta time 
 * 
 * \return double 
 */
static double get_delta_time(void)
{
  static struct timespec t0,t1;

  t0 = t1;
  if(clock_gettime(CLOCK_MONOTONIC,&t1) != 0)
  {
    perror("clock_gettime");
    exit(1);
  }
  return (double)(t1.tv_sec - t0.tv_sec) + 1.0e-9 * (double)(t1.tv_nsec - t0.tv_nsec);
}