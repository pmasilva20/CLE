/**
 *  \file main.c 
 *
 *  \brief Assignment 2 : Problem 2 - Determinant of a Square Matrix
 *
 *  It sorts the list using a binary sorting algorithm which allows the sorting of parts of the original list by
 *  different processes of a group in a hierarchical way.
 *  The list of names must be supplied by the user.
 *  MPI implementation.
 *
 *  \author João Soares 93078 & Pedro Silva 93011
*/

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include "structures.h"
#include <unistd.h>
#include "utils.h"


/**
 * Calculate Matrix Determinant
 * \param size Order of the Matrix
 * \param matrix Matrix
 * \return Determinant of the Matrix after Gaussian Elimination
 */
double calculateMatrixDeterminant(int orderMatrix,double matrix[orderMatrix][orderMatrix]){

    /** \brief Apply Gaussian Elimination
     *  Generic square matrix of order n into an equivalent upper triangular matrix
    */
    for(int i=0;i<orderMatrix-1;i++){
        //Begin Gauss Elimination
        for(int k=i+1;k<orderMatrix;k++){
            double term=matrix[k][i]/matrix[i][i];
            for(int j=0;j<orderMatrix;j++){
                matrix[k][j]=matrix[k][j]-term*matrix[i][j];
            }
        }
    }

    double determinant=1;

    for (int x = 0; x < orderMatrix; x++){
        determinant*= matrix[x][x];
    }

    return determinant;

}


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


    if (rank == 0){ 
        /**
        * \brief Dispatcher process it is the frist process of the group 
        */

        /* Pointer to the text stream associated with the file name */
        FILE *f;              

        /* command */
        unsigned int whatToDo;
        
        unsigned int minProc,                                                /* minimum number of required processes */
                     size,                                                             /* size of processing pattern */
                     n,                                                                         /* counting variable */
                     r;                                                                         /* counting variable */
        unsigned int *ord;                                                                     /* processing pattern */


        /** Initialization FileMatrices*/
        struct FileMatrices file_info;

        /* check running parameters and load list of names into memory */
        if (argc != 2)
           { printf ("No file name!\n");
             whatToDo = NOMOREWORK;
             for (n = 1; n < totProc; n++)
               MPI_Send (&whatToDo, 1, MPI_UNSIGNED, n, 0, MPI_COMM_WORLD);
             MPI_Finalize ();
             return EXIT_FAILURE;
           }

        if ((f = fopen (argv[1], "r")) == NULL)
           { perror ("error on file opening for reading");
             whatToDo = NOMOREWORK;
             for (n = 1; n < totProc; n++)
               MPI_Send (&whatToDo, 1, MPI_UNSIGNED, n, 0, MPI_COMM_WORLD);
             MPI_Finalize ();
             exit (EXIT_FAILURE);
           }

        if(fread(&file_info.numberOfMatrices, sizeof(int), 1, f)==0){
            printf ("Main: Error reading Number of Matrices\n");
        }

        if(fread(&file_info.orderOfMatrices, sizeof(int), 1, f)==0){
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
                printf("Last Worker Value: %d\n",lastWorker);
                for (int n = 1; n< lastWorker+1; n++){
                        struct MatrixResult matrixDeterminantResultReceived;
                        MPI_Recv (&matrixDeterminantResultReceived, sizeof(struct MatrixResult),MPI_BYTE, n, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                        file_info.determinant_result[matrixDeterminantResultReceived.id]=matrixDeterminantResultReceived;
                        
                        printf("Dispatcher %u : Matrix %u Determinant Result. %.3e from worker %d\n",rank,matrixDeterminantResultReceived.id,matrixDeterminantResultReceived.determinant,n);

                        //printf("Reveived Matrix\n");  
                            
                }
            }

            
        }

        fclose(f);
        
        
        whatToDo = NOMOREWORK;
        for (r = 1; r < totProc; r++){
            MPI_Send (&whatToDo, 1, MPI_UNSIGNED, r, 0, MPI_COMM_WORLD);
            printf("Worker %d : Ending\n",r);
        }

        PrintResults(file_info);        
    }


    else { 
        /* Worker Processes the remainder processes of the group*/

        /* command */
        unsigned int whatToDo;                                                                      

        /** Matrix Value */
        struct Matrix val;

        while(true){
            
            MPI_Recv (&whatToDo, 1, MPI_UNSIGNED, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            
            if (whatToDo== NOMOREWORK){
                break;
            }

            /** Matrix Determinant Result */
            struct MatrixResult matrix_determinant_result;
            
            /** REceive Value Matrix */
            MPI_Recv (&val, sizeof (struct Matrix), MPI_BYTE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE); 
            
            //printf("Worker %d : Obtained Matrix %u.\n",rank,val.id);

            matrix_determinant_result.id=val.id;

            /** Calculate Matrix Determinant */
            matrix_determinant_result.determinant=calculateMatrixDeterminant(val.orderMatrix,val.matrix);
    
            //printf("Worker %u : Matrix %u Determinant Result. %.3e \n",rank,val.id,matrix_determinant_result.determinant);
            
            MPI_Send (&matrix_determinant_result,sizeof(struct MatrixResult), MPI_BYTE, 0, 0, MPI_COMM_WORLD);
            //printf("Worker %u : Matrix %u Determinant Obtained.\n",rank,matrix_determinant_result.id);


        }


    }

    MPI_Finalize ();

    return EXIT_SUCCESS;
}        