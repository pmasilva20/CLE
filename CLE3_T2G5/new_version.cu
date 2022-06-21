#include "common.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>


void printResults(char * filename, int numMatrices, int order, double * determinants, double elapsedTime);


__global__ void calcTerms(double *matricesDevice, int *orderDevice, double *determinants, int *currentRow)
{
  int iteration = *currentRow;
  if (threadIdx.x == iteration)
  {
    int order = *orderDevice;
    bool switchedRows = false;
    double *matrix = matricesDevice + blockIdx.x * order * order;

    double pivot =  *(matrix + iteration * order + iteration);
    // finding the pivot
    if (pivot == 0.0)
    {
      for (int k = iteration + 1; k < order; k++)
      {
        if ((matrix + k*order + iteration) != 0)
        {
          // Swap the two rows
          for (int j = 0; j < order; j++)
          {
            double temp = *(matrix + k * order + j);
            *(matrix + k * order + j) = *(matrix + iteration * order + j);
            *(matrix + iteration * order + j) = temp;
          }
          switchedRows = true;
          break;
        }
      }
    }

    pivot = *(matrix + iteration * order + iteration);

    // calculate the determinants
    if (iteration == 0)
      determinants[blockIdx.x] = pivot;
    else
      determinants[blockIdx.x] *= pivot;

    if (switchedRows)
      determinants[blockIdx.x] *= -1;

  }
}

__global__ void subTerms(double *matricesDevice, int *orderDevice, double *determinants, int *currentRow)
{
  int iteration = *currentRow;
  if (threadIdx.x > iteration) {
    int order = *orderDevice;
    double *matrix = matricesDevice + blockIdx.x * order * order;
    double *row = matrix + threadIdx.x * order;
    double *pivotRow = matrix + iteration * order;
    double pivot = *(pivotRow + iteration);

    double scale = row[iteration] / pivot;
    // Begin Gauss Elimination
    for(int k=iteration+1; k<order; k++)
    {
      row[k] -= scale * pivotRow[k];
    }
  }
}

int main(int argc, char **argv)
{
  printf("%s Starting...\n", argv[0]);

  // set up device
  int dev = 0;
  cudaDeviceProp deviceProp;
  CHECK(cudaGetDeviceProperties(&deviceProp, dev));
  printf("Using Device %d: %s\n", dev, deviceProp.name);
  CHECK(cudaSetDevice(dev));

  // getotsps
  char * filename = argv[1];
  FILE *fp = fopen(filename, "r");
  if (fp == NULL)
  {
    printf("Error: could not open file %s\n", filename);
    return EXIT_FAILURE;
  }

  /* get number of matrices in the file */
  int numMatrices;
  if (fread(&numMatrices, sizeof(int), 1, fp) == 0)
  {
    printf("Error: could not read from file %s\n", filename);
    return EXIT_FAILURE;
  }

  /* get order of the matrices in the file */
  int order;
  if (!fread(&order, sizeof(int), 1, fp))
  {
    printf("Error: could not read from file %s\n", filename);
    return EXIT_FAILURE;
  }

  // malloc host memory
  double *matricesHost = (double *)malloc(sizeof(double) * numMatrices * order * order);
  double *determinantsHost = (double *)malloc(sizeof(double) * numMatrices);

  // malloc device global memory the order and all the matrices
  int *orderDevice;
  int *currentRow;
  double *determinants;
  double *matricesDevice;
  CHECK(cudaMalloc((void **)&orderDevice, sizeof(int)));
  CHECK(cudaMalloc((void **)&currentRow, sizeof(int)));
  CHECK(cudaMalloc((void **)&determinants, sizeof(double) * numMatrices));
  CHECK(cudaMalloc((void **)&matricesDevice, sizeof(double) * numMatrices * order * order));

  if (!fread(matricesHost, sizeof(double), numMatrices * order * order, fp))
  {
    printf("Error: could not read from file %s\n", filename);
    return EXIT_FAILURE;
  }

  // transfer data from host to device
  CHECK(cudaMemcpy(orderDevice, &order, sizeof(int), cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(matricesDevice, matricesHost, sizeof(double) * numMatrices*order*order, cudaMemcpyHostToDevice));

  // invoke kernel at host side
  dim3 grid(numMatrices, 1);
  dim3 block(order, 1);

  double iStart = seconds();
  for (int iteration = 0; iteration < order; iteration++)
  {
    // update the currentRow value on the device
    CHECK(cudaMemcpy(currentRow, &iteration, sizeof(int), cudaMemcpyHostToDevice));

    calcTerms<<<grid, block>>>(matricesDevice, orderDevice, determinants, currentRow);
    CHECK(cudaDeviceSynchronize());

    subTerms<<<grid, block>>>(matricesDevice, orderDevice, determinants, currentRow);
    CHECK(cudaDeviceSynchronize());
  }
  double iElaps = seconds() - iStart;

  // check kernel error
  CHECK(cudaGetLastError());

  // copy kernel result back to host side
  CHECK(cudaMemcpy(determinantsHost, determinants, sizeof(double) * numMatrices, cudaMemcpyDeviceToHost));

  // check device results
  printResults(filename, numMatrices, order, determinantsHost, iElaps);

  // free device global memory
  CHECK(cudaFree(orderDevice));
  CHECK(cudaFree(currentRow));
  CHECK(cudaFree(determinants));
  CHECK(cudaFree(matricesDevice));

  // calcular no cpu


  // free host memory
  free(matricesHost);
  free(determinantsHost);

  // reset device
  CHECK(cudaDeviceReset());

  exit(EXIT_SUCCESS);
}

void printResults(char * filename, int numMatrices, int order, double * determinants, double elapsedTime)
{
  printf("\nMatrix File  %s\n", filename);
  printf("Number of Matrices  %d\n", numMatrices);
  printf("Order of the matrices  %d\n", order);

  for (int i=0; i<numMatrices; i++)
  {
    printf("\tMatrix %d Result: Determinant = %.3e \n", i + 1, determinants[i]);
  }

  /* end of measurement */
  printf("\nElapsed time = %.6f s\n", elapsedTime);
}
