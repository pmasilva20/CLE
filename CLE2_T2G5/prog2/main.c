/**
 *  \file main.c 
 *
 *  \brief Assignment 2 : Problem 2 - Determinant of a Square Matrix
 *
 *  Processes the Determinant of Square Matrices
 *  The file with the square matrices must be supplied by the user.
 *  MPI implementation.
 *
 * 
 *  Usage:
 *      \li mpicc -Wall -o main main.c structures.h  utils.c utils.h
 *      \li mpiexec -n <number_processes> ./main <file_name>
 *      \li Example: mpiexec -n 5 ./main mat128_32.bin
 * 
 *  \author João Soares (93078) & Pedro Silva (93011)
*/

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include "structures.h"
#include <unistd.h>
#include "utils.h"
#include <time.h>

/* General definitions */

# define  WORKTODO       1
# define  NOMOREWORK     0

/**
 * \brief Main Function
 *  
 * Instantiation of the processing configuration.
 * 
 * \param argc number of words of the command line
 * \param argv list of words of the command line
 * \return status of operation 
 */

int main (int argc, char *argv[]) {


    /*Number of processes in the Group*/
    int rank; 

    /* Group size */
    int totProc;

    //TODO: Adicionar verificação n workers adicionar comentários, mudar nome de variáveis e alguns prints.
    //TOD0: Testar com 512 e separar funções 

    MPI_Init (&argc, &argv);
    MPI_Comm_rank (MPI_COMM_WORLD, &rank);
    MPI_Comm_size (MPI_COMM_WORLD, &totProc);

    /*processing*/

    if (totProc < 2) {
        printf("Requires at least two processes - one worker.\n");
        MPI_Finalize ();
        return EXIT_FAILURE;
    }

    if (rank == 0){ 
        /**
        * \brief Dispatcher process it is the frist process of the group 
        */

        /** time limits **/
        struct timespec start, finish;

        /* Pointer to the text stream associated with the file name */
        FILE *f;              

        /* command */
        unsigned int whatToDo;

        /* Counting variable */
        unsigned int n;
        
        /* Counting variable */                                                                         
        unsigned int r;                                                                         

        printf("Number of Worker: %d \n",totProc-1);

        /** Initialization FileMatrices*/
        struct FileMatrices file_info;
        
        /** Begin of Time measurement */
        clock_gettime (CLOCK_MONOTONIC_RAW, &start);

        /* Check running parameters and load name into memory */
        if (argc != 2)
           { 
            printf ("No file name!\n");
            whatToDo = NOMOREWORK;
            for (n = 1; n < totProc; n++)
               MPI_Send (&whatToDo, 1, MPI_UNSIGNED, n, 0, MPI_COMM_WORLD);
            MPI_Finalize ();
            return EXIT_FAILURE;
           }

        if ((f = fopen (argv[1], "r")) == NULL)
           { 
            perror ("error on file opening for reading");
            whatToDo = NOMOREWORK;
            for (n = 1; n < totProc; n++)
               MPI_Send (&whatToDo, 1, MPI_UNSIGNED, n, 0, MPI_COMM_WORLD);
             MPI_Finalize ();
             exit (EXIT_FAILURE);
           }

        if (fread(&file_info.numberOfMatrices, sizeof(int), 1, f)==0){
            printf ("Main: Error reading Number of Matrices\n");
        }

        if (fread(&file_info.orderOfMatrices, sizeof(int), 1, f)==0){
            printf("Main: Error reading Order of Matrices\n");
        }

        printf("\nFile %u - Number of Matrices to be read  = %d\n", file_info.id, file_info.numberOfMatrices);

        printf("File %u - Order of Matrices to be read  = %d\n", file_info.id, file_info.orderOfMatrices);

        file_info.determinant_result = malloc(sizeof(struct MatrixResult) * file_info.numberOfMatrices);

        /* Number of Matrices sent*/
        int numberMatricesSent=0;

        while(numberMatricesSent<file_info.numberOfMatrices){
            
            /** Last Worker to receive work **/
            int lastWorker=0;
            

            for (int n = 1; n < totProc; n++){
                
                if (numberMatricesSent==file_info.numberOfMatrices){
                    break;
                }

                struct Matrix matrix1;

                matrix1.fileid = file_info.id;
                matrix1.id = numberMatricesSent;
                matrix1.orderMatrix = file_info.orderOfMatrices;

                if(fread(&matrix1.matrix, sizeof(double), file_info.orderOfMatrices * file_info.orderOfMatrices, f)==0){
                    perror("Main: Error reading Matrix\n");
                }
                
                whatToDo=WORKTODO;
                lastWorker=n;
                
                MPI_Send (&whatToDo, 1, MPI_UNSIGNED, n, 0, MPI_COMM_WORLD);
                
                MPI_Send (&matrix1, sizeof (struct Matrix), MPI_BYTE, n, 0, MPI_COMM_WORLD);
                
                printf("Matrix Processed -> %d to Worker %d \n",numberMatricesSent,n);
                
                numberMatricesSent++;
                
            }
            


            if (lastWorker>0){
                //printf("Last Worker Value: %d\n",lastWorker);
                for (int n = 1; n< lastWorker+1; n++){
                        
                        struct MatrixResult matrixDeterminantResultReceived;
                        
                        MPI_Recv (&matrixDeterminantResultReceived, sizeof(struct MatrixResult),MPI_BYTE, n, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                        file_info.determinant_result[matrixDeterminantResultReceived.id]=matrixDeterminantResultReceived;
                        
                        printf("Dispatcher %u : Matrix %u Determinant Result from worker %d\n",rank,matrixDeterminantResultReceived.id,n);

                        //printf("Reveived Matrix\n");  
                            
                }
            }

            
        }

        /** CLose File */
        fclose(f);

        /** Inform Workers that there is no more work - Workers will end */
        whatToDo = NOMOREWORK;
        for (r = 1; r < totProc; r++){
            MPI_Send (&whatToDo, 1, MPI_UNSIGNED, r, 0, MPI_COMM_WORLD);
            printf("Worker %d : Ending\n",r);
        }

        /** End of measurement */
        clock_gettime (CLOCK_MONOTONIC_RAW, &finish);

        /** Print Final Results */
        printResults(file_info);
        
        /** Print Elapsed Time */
        printf ("\nElapsed time = %.6f s\n",  (finish.tv_sec - start.tv_sec) / 1.0 + (finish.tv_nsec - start.tv_nsec) / 1000000000.0);        
    }


    else { 
        /** \brief  Worker Processes the remainder processes of the group **/

        /* command */
        unsigned int whatToDo;                                                                      

        /** Matrix Value */
        struct Matrix val;

        while(true){

            /** Receive indication of existance of work or not */
            MPI_Recv (&whatToDo, 1, MPI_UNSIGNED, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            
            /** If no more work, worker terminates */
            if (whatToDo== NOMOREWORK){
                break;
            }

            /** Matrix Determinant Result */
            struct MatrixResult matrix_determinant_result;
            
            /** Receive Value Matrix */
            MPI_Recv (&val, sizeof (struct Matrix), MPI_BYTE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE); 
            
            //printf("Worker %d : Obtained Matrix %u.\n",rank,val.id);

            matrix_determinant_result.id=val.id;

            /** Calculate Matrix Determinant */
            matrix_determinant_result.determinant=calculateMatrixDeterminant(val.orderMatrix,val.matrix);
                
            /** Send Partial Result of a Matrix Result **/
            MPI_Send (&matrix_determinant_result,sizeof(struct MatrixResult), MPI_BYTE, 0, 0, MPI_COMM_WORLD);
            //printf("Worker %u : Matrix %u Determinant Obtained.\n",rank,matrix_determinant_result.id);


        }

    }

    MPI_Finalize ();

    return EXIT_SUCCESS;
}        