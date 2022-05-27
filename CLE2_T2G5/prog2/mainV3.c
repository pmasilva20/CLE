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
#include <pthread.h>
#include <time.h>
#include "sharedRegion.h"
#include "probConst.h"

/** General definitions */

# define  WORKTODO       1
# define  NOMOREWORK     0

/** Number of processes in the Group */
int rank; 


/** \brief consumer threads return status array */
int *statusWorks;

/** \brief worker life cycle routine */
static void *threadReadFile (void *id);

static void *threadSendMatrices (void *id);

/** Group size */
int totProc;

/** Initialization FileMatrices*/
struct FileMatrices file_info;

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




    // Initilialise MPI and ask for thread support
    int provided;

    statusWorks = malloc(sizeof(int)*1);

    pthread_t tIdWorkers[2];

    unsigned int works[2];
    /** Pointer to execution status */
    int *status_p;
    
    for (int i = 0; i < 2; i++)
        works[i] = i;

    srandom ((unsigned int) getpid ());
    
    MPI_Init_thread (&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    MPI_Comm_rank (MPI_COMM_WORLD, &rank);
    MPI_Comm_size (MPI_COMM_WORLD, &totProc);

    if(provided < MPI_THREAD_MULTIPLE)
    {
        printf("The threading support level is lesser than that demanded.\n");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
    else
    {
        printf("The threading support level corresponds to that demanded.\n");
    }

    /*processing*/
    /** Verify if there is enough processes **/
    if (totProc < 2) {
        printf("Requires at least two processes - one worker.\n");
        MPI_Finalize ();
        return EXIT_FAILURE;
    }

    if (rank == 0){ 
        /**
        * \brief Dispatcher process it is the first process of the group
        */

        /** time limits **/
        struct timespec start, finish;
       
        /** Variable to command if there is work to do or not  */
        unsigned int whatToDo;

        printf("Number of Worker: %d \n",totProc-1);
        
        /** Begin of Time measurement */
        clock_gettime (CLOCK_MONOTONIC_RAW, &start);

        /** Check running parameters and load name into memory */
        if (argc != 2)
           { printf ("No file name!\n");
             whatToDo = NOMOREWORK;
             for (int n = 1; n < totProc; n++)
               MPI_Send (&whatToDo, 1, MPI_UNSIGNED, n, 0, MPI_COMM_WORLD);
             MPI_Finalize ();
             return EXIT_FAILURE;
           }

        if ((file_info.pFile = fopen (argv[1], "r")) == NULL)
           { perror ("error on file opening for reading");
             whatToDo = NOMOREWORK;
             for (int n = 1; n < totProc; n++)
               MPI_Send (&whatToDo, 1, MPI_UNSIGNED, n, 0, MPI_COMM_WORLD);
             MPI_Finalize ();
             exit (EXIT_FAILURE);
           }

        if (fread(&file_info.numberOfMatrices, sizeof(int), 1, file_info.pFile)==0){
            printf ("Main: Error reading Number of Matrices\n");
        }

        if (fread(&file_info.orderOfMatrices, sizeof(int), 1, file_info.pFile)==0){
            printf("Main: Error reading Order of Matrices\n");
        }

        printf("\nFile %u - Number of Matrices to be read  = %d\n", file_info.id, file_info.numberOfMatrices);

        printf("File %u - Order of Matrices to be read  = %d\n", file_info.id, file_info.orderOfMatrices);

        /** Set File Properties */
        file_info.determinant_result = malloc(sizeof(struct MatrixResult) * file_info.numberOfMatrices);
        
        strcpy(file_info.name, argv[1]);
        
        /* Number of Matrices sent*/
        int numberMatricesSent=0;

        //TODO: METER DOCUMENTAÇÃO


        struct MatrixResult *recData;

        struct Matrix *senData;
                
        bool allMsgRec, recVal, msgRec[totProc];
        
        MPI_Request reqSnd[totProc], reqRec[totProc];

        recData=(struct MatrixResult*)malloc(sizeof(struct MatrixResult) * totProc );
        
        senData=(struct Matrix*)malloc(sizeof(struct Matrix) * file_info.numberOfMatrices );

        if (pthread_create(&tIdWorkers[0], NULL, threadReadFile, &works[0]) !=0)
            {
                perror("error on creating thread worker");
                exit(EXIT_FAILURE);
            }
            else{
                printf("Thread Worker Created !\n");
            }
        

        if (pthread_create(&tIdWorkers[1], NULL, threadSendMatrices, &works[1]) !=0)
            {
                perror("error on creating thread worker");
                exit(EXIT_FAILURE);
            }
            else{
                printf("Thread Worker Created!\n");
            }
        


        int numberMatricesReceived=0;

        while(numberMatricesReceived<file_info.numberOfMatrices){
            
            
            /** If lastWorker is > 0, it means that the dispatcher previously sent work to a certain number of workers
             *  else it means that it didn't send work so it will not receive any partial result
             * */
            for (int n = (rank + 1) % totProc; n< totProc; n++){                        
                    MPI_Irecv (&recData[n],sizeof (struct MatrixResult),MPI_BYTE, n, 0, MPI_COMM_WORLD, &reqRec[n]);                        
                    msgRec[n] = false;
            }
        
            do
            { allMsgRec = true;
                for (int i = (rank + 1) % totProc; i < totProc; i++){
                    if (!msgRec[i]){ 
                        recVal = false;
                        MPI_Test(&reqRec[i], (int *) &recVal, MPI_STATUS_IGNORE);
                        if (recVal){
                            printf("Dispatcher %u : Matrix %u Determinant Result from worker %d\n",rank,recData[i].id,i);
                            file_info.determinant_result[recData[i].id]=recData[i];
                            numberMatricesReceived++;
                            msgRec[i] = true;
                            }
                            else allMsgRec = false;
                    }
                }
            } while (!allMsgRec);
        }
    

        /** Close File */
        fclose(file_info.pFile);

        /** Inform Workers that there is no more work - Workers will end */
        whatToDo = NOMOREWORK;
        for (int r = 1; r < totProc; r++){
            MPI_Send (&whatToDo, 1, MPI_UNSIGNED, r, 0, MPI_COMM_WORLD);
            printf("Worker %d : Ending\n",r);
        }

            for (int i = 0; i < 2; i++)
            { if (pthread_join (tIdWorkers[i], (void *) &status_p) != 0)
                {
                    fprintf(stderr, "Worker %u : error on waiting for thread producer ",i);
                    perror("");
                    exit (EXIT_FAILURE);
                }
                else{
                    printf ("Worker %u : has terminated with status: %d \n", i, *status_p);

                }
            }

        /** End of measurement */
        clock_gettime (CLOCK_MONOTONIC_RAW, &finish);

        /** Print Final Results */
        printResults(file_info);
        
        /** Print Elapsed Time */
        printf ("\nElapsed time = %.6f s\n",  (finish.tv_sec - start.tv_sec) / 1.0 + (finish.tv_nsec - start.tv_nsec) / 1000000000.0);        

        //pthread_join(tIdWorkers,NULL);
    }


    else { 
        /** \brief  Worker Processes the remainder processes of the group **/

        /** Variable to command if there is work to do or not
         *  If the message indicates no work, the Worker will End
         * */
        unsigned int whatToDo;                                                                      

        /** Matrix Value */
        struct Matrix val;
        
        //TODO: METER DOCUMENTAÇÃO

        MPI_Request requestMatrix;

        MPI_Request requestWhatToDo;
        
        bool recValMatrix;

        bool recValWhatToDo;
        
        while(true){

            /** Receive indication of existence of work or not */            
            MPI_Irecv (&whatToDo, 1, MPI_UNSIGNED, 0, 0, MPI_COMM_WORLD, &requestWhatToDo);
      
            recValMatrix=false;

            while (!recValWhatToDo){
                MPI_Test(&requestWhatToDo, (int *) &recValWhatToDo, MPI_STATUS_IGNORE);
            }
            
            /** If no more work, worker terminates */
            if (whatToDo== NOMOREWORK){
                break;
            }

            /** Matrix Determinant Result */
            struct MatrixResult matrix_determinant_result;
            
            /** Receive Value Matrix */
            MPI_Irecv (&val, sizeof (struct Matrix), MPI_BYTE, 0, 0, MPI_COMM_WORLD, &requestMatrix); 
            
            recValMatrix=false;

            while (!recValMatrix){
                MPI_Test(&requestMatrix, (int *) &recValMatrix, MPI_STATUS_IGNORE);
            }
            
            matrix_determinant_result.id=val.id;

            /** Calculate Matrix Determinant */
            matrix_determinant_result.determinant=calculateMatrixDeterminant(val.orderMatrix,val.matrix);
                
            /** Send Partial Result of a Matrix Result **/
            MPI_Send (&matrix_determinant_result,sizeof(struct MatrixResult), MPI_BYTE, 0, 0, MPI_COMM_WORLD);

        }

    }

    MPI_Finalize ();

    return EXIT_SUCCESS;
}        

static void *threadReadFile (void *par)
{
    /** Worker ID */
    unsigned int id = *((unsigned int *) par);

    int numberMatricesProcessed=0;

    

    while(numberMatricesProcessed<file_info.numberOfMatrices){
        
        struct Matrix val;
        
        val.fileid=file_info.id;
        
        val.id=numberMatricesProcessed;
        
        val.orderMatrix=file_info.orderOfMatrices;
        
        if(fread(&val.matrix, sizeof(double), file_info.orderOfMatrices * file_info.orderOfMatrices, file_info.pFile)==0){
            perror("Main: Error reading Matrix\n");
        }

        putMatrixVal(id,val);
        //Enviar para o FIFO
        numberMatricesProcessed++;
    
    }

    printf("Hello World! %d\n",totProc);

    statusWorks[id] = EXIT_SUCCESS;
    pthread_exit (&statusWorks[id]);

}

static void *threadSendMatrices (void *par){

    unsigned int id = *((unsigned int *) par);

    int numberMatricesSent=0;

    /** Variable to command if there is work to do or not  */
    unsigned int whatToDo;

    struct Matrix *senData;
    
    senData=(struct Matrix*)malloc(sizeof(struct Matrix) * file_info.numberOfMatrices );
    
    MPI_Request reqSnd[totProc];

    while (numberMatricesSent<file_info.numberOfMatrices){
        MPI_Request request;
        for (int n = (rank + 1) % totProc; n < totProc; n++){

            if (numberMatricesSent==file_info.numberOfMatrices){
                break;
            }
            
            getMatrixVal(&senData[n]);
            
            
            /**There is Work to do **/
            whatToDo=WORKTODO;

            MPI_Isend (&whatToDo, 1, MPI_UNSIGNED, n, 0, MPI_COMM_WORLD,&request);

            MPI_Isend (&senData[n], sizeof (struct Matrix), MPI_BYTE, n, 0, MPI_COMM_WORLD,&reqSnd[n]);
            
            printf("Matrix Processed -> Matrix %d to Worker %d \n",numberMatricesSent,n);

            /** Update Number of work sent*/
            numberMatricesSent++;
            
        }
        
    }
    
    statusWorks[id] = EXIT_SUCCESS;
    pthread_exit (&statusWorks[id]);
    
}
