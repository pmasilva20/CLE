#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

int main (int argc, char *argv[]){
    int rank, size;
    int i = 0;
    int *sendData = NULL, *recData;
    //int *reduce1Data = NULL;
    int val;

    //int maxVal1;


    //int maxVal, minVal;
    
    MPI_Init (&argc, &argv);
    MPI_Comm_rank (MPI_COMM_WORLD, &rank);
    MPI_Comm_size (MPI_COMM_WORLD, &size);
    srandom (getpid ());


    //scatter1
    recData = malloc (4 * sizeof (int));
    if (rank == 0){ 
        //Process 0 sends data to all, doing 16
        sendData = malloc (16 * size * sizeof (int));
        printf ("Data to be scattered\n");
        for (int i = 0; i < 16 * size; i++){
            val = ((double) rand () / RAND_MAX) * 1000;
            sendData[i] = val;
            printf ("%d ", sendData[i]);
        }
        printf ("\n");
    }
    MPI_Scatter (sendData, 4, MPI_INT, recData, 4, MPI_INT, 0, MPI_COMM_WORLD);

    printf ("Received data after Scatter 1 by process %d: ", rank);
    for (i = 0; i < 4; i++)printf ("%2d ", recData[i]);
    printf ("\n");




    // //reduce1
    // reduce1Data = malloc (4 * sizeof (int));

    // //MPI_Reduce (&val, &minVal, 1, MPI_INT, MPI_MIN, 0, MPI_COMM_WORLD);

    // MPI_Reduce (recData, reduce1Data, 4, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);

    // if(rank == 0){
    //     printf ("Data after reduce by process %d: ", rank);
    //      for (i = 0; i < 4; i++)printf ("%2d ", reduce1Data[i]);
    //     printf("\n");
    // }







    //scatter2

    // int recData2 = 0;
    // MPI_Scatter (reduce1Data, 4, MPI_INT, &recData2, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // printf ("Received data after Scatter 2 by process %d: %d \n", rank,recData2);


    //final reduce pra min e max


    // MPI_Reduce (&val, &minVal, 1, MPI_INT, MPI_MIN, 0, MPI_COMM_WORLD);
    // MPI_Reduce (&val, &maxVal, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
    
    // if (rank == 0){
    //     printf ("Largest value = %d\n", maxVal);
    //     printf ("Smallest value = %d\n", minVal);
    // }
    
    MPI_Finalize ();
    return EXIT_SUCCESS;
}