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


  if(rank == 0){
    char data[] = "I'm 0 and I am alive and well!";
    char* recData = data;
    for(int i = 1; i < size; i++){
      //Send to everyone a message
      printf ("Rank 0 Transmitted message: %s to rank %d \n", data,i);
      MPI_Send (data, strlen (data), MPI_CHAR, i, 0, MPI_COMM_WORLD);

    }
    for(int i = 1; i < size; i++){
      //Receive message from all others stating they are alive
      recData = malloc (100);
      for (int i = 0; i < 100; i++)recData[i] = '\0';

      MPI_Recv (recData, 100, MPI_CHAR, i, 0 , MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      printf ("Rank 0 Received message: %s from %d \n", recData,i);
    }
  }
  else{
      char data[] = "I am also alive and well!";
      char* recData = data;

      //Receive message from 0
      recData = malloc (100);
      for (int i = 0; i < 100; i++)recData[i] = '\0';

      MPI_Recv (recData, 100, MPI_CHAR, 0, 0 , MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      printf ("Rank %d Received message: %s from %d \n", rank,recData,0);
      //Respond to 0
      printf ("Rank %d Transmitted message: %s to rank %d \n", rank,data,0);
      MPI_Send (data, strlen (data), MPI_CHAR, 0, 0, MPI_COMM_WORLD);
  }


  MPI_Finalize ();

  return EXIT_SUCCESS;
}
