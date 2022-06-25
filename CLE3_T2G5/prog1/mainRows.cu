/**
 *   
 */

#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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

/* allusion to internal functions */

static void modify_sector_cpu_kernel (unsigned int *sector_data, unsigned int sector_number, unsigned int n_sectors,
                                      unsigned int sector_size);
__global__ static void modify_sector_cuda_kernel (unsigned int * __restrict__ sector_data, unsigned int * __restrict__ sector_number,
                                                  unsigned int n_sectors, unsigned int sector_size);

__global__  static void updateMatrixCols(double* d_matrices, double* d_results);

static double get_delta_time(void);
void printResults(int numberOfMatrices, double * resultsDeterminant);
/**
 *   main program
 */

int main (int argc, char **argv)
{
  //printf("%s Starting...\n", argv[0]);
  //if (sizeof (unsigned int) != (size_t) 4)
     //return 1;                                             // it fails with prejudice if an integer does not have 4 bytes

  
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
  (void) get_delta_time ();
  updateMatrixCols <<<grid, block>>> (matricesDevice,resultsDeterminantDevice);
  CHECK (cudaDeviceSynchronize ());                            // wait for kernel to finish
  CHECK (cudaGetLastError ());                                 // check for kernel errors
  printf("The CUDA kernel <<<(%d,%d,%d), (%d,%d,%d)>>> took %.3e seconds to run\n",
         gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, get_delta_time ());

  /* Copy kernel result back to host side */

  /** Determinant Results from Device/GPU **/
  double *resultsDeterminantFromDevice = (double *)malloc(sizeof(double) * numberOfMatrices);

  CHECK(cudaMemcpy(resultsDeterminantFromDevice, resultsDeterminantDevice, numberOfMatrices * sizeof(double), cudaMemcpyDeviceToHost));
  printf ("The transfer of Determinant Results from the device to the host took %.3e seconds\n", get_delta_time ());

  /** Free Matrices from Device/GPU  Global Memory **/
  CHECK (cudaFree (matricesDevice));

  /** Free Results Determinant Device/GPU Global Memory **/
  CHECK (cudaFree (resultsDeterminantDevice));

  /* Reset the Device / GPU */
  CHECK (cudaDeviceReset ());

  printResults(numberOfMatrices,resultsDeterminantFromDevice);
  
  return 0; 
}

static void modify_sector_cpu_kernel (unsigned int *sector_data, unsigned int sector_number, unsigned int n_sectors,
                                      unsigned int sector_size)
{
  unsigned int x, i, a, c, n_words;

  /* convert the sector size into number of 4-byte words (it is assumed that sizeof(unsigned int) = 4) */

  n_words = sector_size / 4u;

  /* initialize the linear congruencial pseudo-random number generator
     (section 3.2.1.2 of The Art of Computer Programming presents the theory behind the restrictions on a and c) */

  i = sector_number;                                       // get the sector number
  a = 0xCCE00001u ^ ((i & 0x0F0F0F0Fu) << 2);              // a must be a multiple of 4 plus 1
  c = 0x00CCE001u ^ ((i & 0xF0F0F0F0u) >> 3);              // c must be odd
  x = 0xCCE02021u;                                         // initial state

  /* modify the sector data */

  for (i = 0u; i < n_words; i++)
  { x = a * x + c;                                         // update the pseudo-random generator state
    sector_data[i] ^= x;                                   // modify the sector data
  }
}

//TODO: Colocar aqui versão CPU

__global__  static void updateMatrixCols(double* d_matrices, double* d_results) {
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

void printResults(int numberOfMatrices, double *resultsDeterminant){
  
  for (int a = 0; a < numberOfMatrices; a++) {
      printf("Matrix %d :\n", a + 1);
      printf("The determinant is %.3e\n", resultsDeterminant[a]);
  }
}