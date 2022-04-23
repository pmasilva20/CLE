/**
 *  \file sharedRegion.c
 *
 *  \brief Assignment 1 : Problem 2 - Determinant of a Square Matrix
 * *
 *  Shared Region
 *
 *  Main Operations:
 *     \li putFileInfo
 *     \li putMatrixVal
 *     \li PrintResults
 *
 *  Workers Operations:
 *     \li getMatrixVal
 *     \li putResults
 *
 *
 *  \author João Soares (93078) e Pedro Silva (93011)
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <unistd.h>
#include <pthread.h>
#include <errno.h>
#include "probConst.h"
#include "structures.h"

/** \brief consumer threads return status array */
extern int *statusWorks;

/** \brief Number of matrices already processed */
extern int matrixProcessed;

/** \brief  Number of Matrices to be processed by workers **/
int matrixToProcess;

/** \brief Number of matrices sent to SharedRegion */
static unsigned int nMatricesSentToSharedRegion;

/** \brief Number of workers waiting to access Stored Region of Matrices */
static unsigned int nWorkersWaiting;

/** \brief Number of matrices available in Shared Region */
static unsigned int nMatricesInSharedRegion;

/** \brief Matrices storage region */
static struct Matrix matrix_mem[K];

/** \brief Files storage region */
static struct FileMatrices file_mem[N];

/** \brief insertion pointer for file_mem */
static unsigned int  ii_fileInfo;

/** \brief insertion pointer for matrix_mem */
static unsigned int  ii_matrix;

/** \brief insertion pointer for matrix_mem */
static unsigned int  ri_matrix;

/** \brief flag signaling the data transfer region Matrix is full */
static bool fullMatrixMem;

/** \brief producers synchronization point when the data transfer region is full */
static pthread_cond_t fifoMatrixFull;

/** \brief consumers synchronization point when the data transfer region is empty */
static pthread_cond_t fifoMatrixEmpty;

/** \brief locking flag which warrants mutual exclusion inside the monitor */
static pthread_mutex_t accessCR = PTHREAD_MUTEX_INITIALIZER;

/** \brief flag which warrants that the data transfer region is initialized exactly once */
static pthread_once_t init = PTHREAD_ONCE_INIT;;

/**
 *  \brief Initialization of the data transfer region.
 *
 *  Internal monitor operation.
 */
static void initialization (void)
{

    ii_matrix = ri_matrix = nWorkersWaiting = 0;
    fullMatrixMem = false;

    /* initialize of synchronization points */
  pthread_cond_init (&fifoMatrixFull, NULL);
  pthread_cond_init (&fifoMatrixEmpty, NULL);
}

/**
 *  \brief Store a File (FileMatrices) value in the data transfer region.
 *
 *  Operation carried out by the Main.
 *
 *  \param val File (FileMatrices) to be stored
 */
void putFileInfo(struct FileMatrices fileInfo){

    /** Enter Monitor */
    if(pthread_mutex_lock (&accessCR)!=0){
        printf("Main: error on entering monitor(CF)");
    }

    /** Close pFile if not null */
    if(file_mem[ii_fileInfo].pFile!=NULL){
        fclose(file_mem[ii_fileInfo].pFile);
    }

    /** Update Number of Matrices to Process */
    matrixToProcess += fileInfo.numberOfMatrices;

    /** Save File in Shared Region */
    file_mem[ii_fileInfo] = fileInfo;

    ii_fileInfo++;

    /** Exit Monitor */
    if(pthread_mutex_unlock (&accessCR)!=0){
        printf("Main: error on exiting monitor(CF)");
    }

}

/**
 *  \brief Store a Matrix value in the data transfer region.
 *
 *  Operation carried out by the workers.
 *
 *  \param prodId worker identification
 *  \param val Matrix to be stored
 */
void putMatrixVal(struct Matrix matrix){

    /** Enter Monitor */
    if(pthread_mutex_lock (&accessCR)!=0){
        printf("Main: error on entering monitor(CF)");
    }

    pthread_once (&init, initialization);


    /** While FIFO full wait */
    while (fullMatrixMem){
        if(pthread_cond_wait (&fifoMatrixFull, &accessCR)!=0){
            printf("Main: error on waiting in fifoFull");
        }
    };

    matrix_mem[ii_matrix]= matrix;
    ii_matrix= (ii_matrix+1)%K;

    fullMatrixMem = (ii_matrix == ri_matrix);

    /** Increase Number of Matrices sent to Shared Region */
    nMatricesSentToSharedRegion++;

    /** Increase Number of Matrices in Shared Region */
    nMatricesInSharedRegion++;

    /** Let a Worker know that a Matrix has been stored */
    if(pthread_cond_signal (&fifoMatrixEmpty)!=0){
        printf("Main: error on signaling in fifoEmpty");
    }

    /** Exit monitor */
    if(pthread_mutex_unlock (&accessCR)!=0){
        printf("Main: error on exiting monitor(CF)");
    }

}


/**
 *  \brief Get a Matrix value from the data transfer region.
 *
 *  Operation carried out by the workers.
 *
 *  \param consId worker identification
 *
 *  \return value
 */
int getMatrixVal(unsigned int consId,struct Matrix *matrix)
{
    /** Enter Monitor */
    if ((statusWorks[consId] = pthread_mutex_lock (&accessCR)) != 0)
    {errno = statusWorks[consId];
        perror ("error on entering monitor(CF)");
        statusWorks[consId] = EXIT_FAILURE;
        pthread_exit (&statusWorks[consId]);
    }

    pthread_once (&init, initialization);

    while ((ii_matrix == ri_matrix) && !fullMatrixMem)
    {
        /**
         * If the number of matrices processed is equal to the number of matrices to be processed then the worker knows that it has run out of work.
         *
         * Or if there is no work available for the waiting workers, that is, if the number of waiting workers is greater than the number of matrices
         * in the shared region and if the difference between the matrices to process and matrices already sent to the shared
         * region is equal or less than the number of workers waiting.
         */
        if((matrixProcessed==matrixToProcess) || (nWorkersWaiting > nMatricesInSharedRegion && (matrixToProcess - nMatricesSentToSharedRegion <= nWorkersWaiting))){

            if ((statusWorks[consId] = pthread_mutex_unlock (&accessCR)) != 0)
            { errno = statusWorks[consId];
                perror ("error on exiting monitor(CF)");
                statusWorks[consId] = EXIT_FAILURE;
                pthread_exit (&statusWorks[consId]);
            }

            /** There is not enough Work for the Worker*/
            return 1;
        }
        /** Increase Number of Workers waiting */
        nWorkersWaiting++;

        /** Exit monitor */
        if ((statusWorks[consId] = pthread_cond_wait (&fifoMatrixEmpty, &accessCR)) != 0)
        { errno = statusWorks[consId];
            perror ("error on waiting in fifoEmpty");
            statusWorks[consId] = EXIT_FAILURE;
            pthread_exit (&statusWorks[consId]);
        }

        /** Worker no longer waiting - Decrease Number of Workers waiting */
        nWorkersWaiting--;
    }

    *matrix = matrix_mem[ri_matrix];

    ri_matrix = (ri_matrix + 1) % K;

    fullMatrixMem = false;

    /** Increase Number of Matrices processed */
    matrixProcessed++;

    /** Decrease Number of Matrices in Shared Region */
    nMatricesInSharedRegion--;

    /** Let Main know that a Matrix has been retrieved */
    if ((statusWorks[consId] = pthread_cond_signal (&fifoMatrixFull)) != 0)
    { errno = statusWorks[consId];
        perror ("error on signaling in fifoFull");
        statusWorks[consId] = EXIT_FAILURE;
        pthread_exit (&statusWorks[consId]);
    }

    /** Exit monitor */
    if ((statusWorks[consId] = pthread_mutex_unlock (&accessCR)) != 0)
    { errno = statusWorks[consId];
        perror ("error on exiting monitor(CF)");
        statusWorks[consId] = EXIT_FAILURE;
        pthread_exit (&statusWorks[consId]);
    }

    /** There is Work to do*/
    return 0;
}


/**
 *  \brief Store a Determinant value of Matrix in the data transfer region.
 *
 *  Operation carried out by the workers.
 *
 *  \param prodId worker identification
 *  \param val Determinant value of Matrix to be stored
 */
void putResults(struct MatrixResult result, unsigned int consId){

    /** Enter Monitor */
    if ((statusWorks[consId] = pthread_mutex_lock (&accessCR)) != 0)
    { errno = statusWorks[consId];
        perror ("error on entering monitor(CF)");
        statusWorks[consId] = EXIT_FAILURE;
        pthread_exit (&statusWorks[consId]);
    }

    pthread_once (&init, initialization);

    /** Save Results in Shared Region*/
    for (int x = 0; x < ii_fileInfo; x++){
        if(file_mem[x].id==result.fileid){
            int id =result.id;
            file_mem[x].determinant_result[id]=result;
            break;
        }
    }

    /** Exit monitor */
    if ((statusWorks[consId] = pthread_mutex_unlock (&accessCR)) != 0)
    { errno = statusWorks[consId];
        perror ("error on exiting monitor(CF)");
        statusWorks[consId] = EXIT_FAILURE;
        pthread_exit (&statusWorks[consId]);
    }
}

/**
 * Print in the terminal the results stored in the Shared Region
 * \param filesToProcess Number of Files
 */
void PrintResults(int filesToProcess){
    for (int x = 0; x < filesToProcess; x++){
        printf("\nFile: %s\n",file_mem[x].name);
        if(file_mem[x].numberOfMatrices) {
            for (int a = 0; a < file_mem[x].numberOfMatrices; a++) {
                printf("Matrix %d :\n", file_mem[x].determinant_result[a].id + 1);
                printf("The determinant is %.3e\n", file_mem[x].determinant_result[a].determinant);
            }

            /** Free allocated memory */
            free(file_mem[x].determinant_result);
        }
        else{
            printf("Error Reading File\n");
        }
        printf("\n");

    }
}


