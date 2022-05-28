#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main (int argc, char *argv[])
{
   int rank, size;

  //char data[] = "I am here!",*recData;

  MPI_Init (&argc, &argv);
  MPI_Comm_rank (MPI_COMM_WORLD, &rank);
  MPI_Comm_size (MPI_COMM_WORLD, &size);

  // Allocates storage
  char* data = (char*)malloc(11 * sizeof(char));
  char* recData = data;
  sprintf(data, "I am here %d!", rank);


  int previous = rank-1;
  if(previous < 0){
    previous = size-1;
  }

  //First one in ring
  if(rank == 0){
    //0 starts out by sending message to 1, and waits for message from size-1
    printf ("Rank %d transmitted message: %s to %d\n",rank ,data, (rank+1) % size);
    MPI_Send (data, strlen (data), MPI_CHAR, (rank+1) % size, 0, MPI_COMM_WORLD);

    MPI_Recv (recData, 100, MPI_CHAR, previous, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    printf ("Rank %d received message: %s from %d \n",rank ,recData, previous);

  }
  //Receives from previous, sends to next one
  else{
    recData = malloc (100);
    for (int i = 0; i < 100; i++)recData[i] = '\0';

    MPI_Recv (recData, 100, MPI_CHAR, previous, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    printf ("Rank %d received message: %s from %d \n",rank ,recData, previous);

    printf ("Rank %d transmitted message: %s to %d\n",rank ,data, (rank+1) % size);
    MPI_Send (data, strlen (data), MPI_CHAR, (rank+1) % size, 0, MPI_COMM_WORLD);

  }

  MPI_Finalize ();


  return EXIT_SUCCESS;
}
