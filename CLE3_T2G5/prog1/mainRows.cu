/**
 *   
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

__global__  static void computeDeterminantByRowsOnGPU(double* d_matrices, double* d_results);

void calcDeterminantCPU(double* matrices_cpu, double* determinant_cpu, int n, int m);

static double get_delta_time(void);

void printResults(int numberOfMatrices, double * resultsDeterminant);

void compareResults(double* resultsDeterminantDevice, double* resultsDeterminantHost,int numberOfMatrices);


/**
 *   Main program
 */

int main (int argc, char **argv)
{
  printf("%s Starting...\n", argv[0]);
  if (sizeof (unsigned int) != (size_t) 4)
     return 1;                                             // it fails with prejudice if an integer does not have 4 bytes

  
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

  /* run the computational kernel
     as an example, N_SECTORS threads are launched where each thread deals with one sector */

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
  
  computeDeterminantByRowsOnGPU <<<grid, block>>> (matricesDevice,resultsDeterminantDevice);

  CHECK (cudaDeviceSynchronize ());                            // wait for kernel to finish
  CHECK (cudaGetLastError ());                                 // check for kernel errors
  
  double deviceTime=get_delta_time();

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
  
  calcDeterminantCPU(matricesHost,resultsDeterminantHost,orderOfMatrices,numberOfMatrices);
  
  double cpuTime=get_delta_time();
 
  printResults(numberOfMatrices,resultsDeterminantFromDevice);
  
  /** Compare Determinant Results from Host and Device **/
  compareResults(resultsDeterminantFromDevice,resultsDeterminantHost,numberOfMatrices);

  printf("\nThe CPU Kernel took %.3e seconds to run\n",cpuTime);
  printf("The Device CUDA Kernel <<<(%d,%d,%d), (%d,%d,%d)>>> took %.3e seconds to run\n",
         gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, deviceTime);
  
  return 0; 
}

//TODO: Colocar aqui versão CPU alterar formatar
//TODO alterar função E Documentação
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

//TODO alterar função E Documentação
__global__ void static computeDeterminantByRowsOnGPU(double* matricesDevice, double* resultsDeterminantDevice) {
	
  int numberOfMatrices = blockDim.x;
	
  for (int iter = 0; iter < numberOfMatrices; iter++) {
   
    if (threadIdx.x < iter) 
      continue;

    int matrixId = blockIdx.x * blockDim.x * blockDim.x;
    int row = matrixId + threadIdx.x * blockDim.x;			// current row offset of this (block thread)	
    int iterRow = matrixId + iter * blockDim.x;

    if (threadIdx.x == iter) {
      if (iter == 0)
        resultsDeterminantDevice[blockIdx.x] = 1;
      resultsDeterminantDevice[blockIdx.x] *= matricesDevice[iterRow + iter];
      continue;
    }

    double pivot = matricesDevice[iterRow + iter];

    double value = matricesDevice[row + iter] / pivot;
    for (int i = iter + 1; i < numberOfMatrices; i++) {
      matricesDevice[row + i] -= matricesDevice[iterRow + i] * value; 
    }
    __syncthreads();
  }
}

void printResults(int numberOfMatrices, double *resultsDeterminant){
  for (int a = 0; a < numberOfMatrices; a++) {
      printf("Matrix %d :\n", a + 1);
      printf("The determinant is %.3e\n", resultsDeterminant[a]);
  }
}

void compareResults(double* resultsDeterminantDevice, double* resultsDeterminantHost,int numberOfMatrices){

  bool difference=false;
  double epsilon = 0.000001;
  for(int i = 0; i < numberOfMatrices; i++){
    
    if(!(fabs(resultsDeterminantDevice[i]-resultsDeterminantHost[i]) < epsilon)){
      printf("\nResults from Device different compared with Results from Host!\n");
      difference=true;
      break;
    }

  }
  if(!difference){
    printf("\nResults from Device equal to Results from Host!\n");
  }

}



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