/**
 *  \file sharedRegion.c
 *
 *  \brief Assignment 2 : Problem 2 - Determinant of a Square Matrix
 * *
 *  Shared Region
 *
 *  Dispatcher Operations:
 *     \li getMatrixVal
 *     \li putMatrixVal
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
extern int *statusDispatcherThreads;

/** \brief  Number of Matrices to be processed by workers **/
int matrixToProcess;

/** \brief Number of matrices sent to SharedRegion */
static unsigned int nMatricesSentToSharedRegion;

/** \brief Matrices storage region */
static struct Matrix matrix_mem[K];

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

    ii_matrix = ri_matrix = 0;
    fullMatrixMem = false;

  /* initialize of synchronization points */
  pthread_cond_init (&fifoMatrixFull, NULL);
  pthread_cond_init (&fifoMatrixEmpty, NULL);
}

/**
 *  \brief Store a Matrix value in the data transfer region.
 *
 *  Operation carried out by the Thread Read Matrices of the File.
 *
 *  \param consId thread identification
 *  \param val Matrix to be stored
 */
void putMatrixVal(unsigned int consId,struct Matrix matrix){

    /** Enter Monitor */
    if ((statusDispatcherThreads[consId] = pthread_mutex_lock (&accessCR)) != 0)
    {errno = statusDispatcherThreads[consId];
        perror ("error on entering monitor(CF)");
        statusDispatcherThreads[consId] = EXIT_FAILURE;
        pthread_exit (&statusDispatcherThreads[consId]);
    }

    pthread_once (&init, initialization);


    /** While FIFO full wait */
    while (fullMatrixMem){
        if ((statusDispatcherThreads[consId] = pthread_cond_wait (&fifoMatrixFull, &accessCR)) != 0)
            { errno = statusDispatcherThreads[consId];
                perror ("error on waiting in fifoFull");
                statusDispatcherThreads[consId] = EXIT_FAILURE;
                pthread_exit (&statusDispatcherThreads[consId]);
            }
    };

    matrix_mem[ii_matrix]= matrix;
    ii_matrix= (ii_matrix+1)%K;

    fullMatrixMem = (ii_matrix == ri_matrix);

    /** Increase Number of Matrices sent to Shared Region */
    nMatricesSentToSharedRegion++;

    /** Let Main know that a Matrix has been retrieved */
    if ((statusDispatcherThreads[consId] = pthread_cond_signal (&fifoMatrixEmpty)) != 0)
    { errno = statusDispatcherThreads[consId];
        perror ("error on signaling in fifoEmpty");
        statusDispatcherThreads[consId] = EXIT_FAILURE;
        pthread_exit (&statusDispatcherThreads[consId]);
    }

    /** Exit monitor */
    if ((statusDispatcherThreads[consId] = pthread_mutex_unlock (&accessCR)) != 0)
    { errno = statusDispatcherThreads[consId];
        perror ("error on exiting monitor(CF)");
        statusDispatcherThreads[consId] = EXIT_FAILURE;
        pthread_exit (&statusDispatcherThreads[consId]);
    }
}


/**
 *  \brief Get a Matrix value from the data transfer region.
 *
 *  Operation carried out by the Thread Send Matrices and Receive Results.
 *
 *  \param consId thread identification
 *  \param *matrix Address of Variable Matrix
 *
 *  \return Whatever there is Work to do;
 */
int getMatrixVal(unsigned int consId,struct Matrix *matrix)
{
    /** Enter Monitor */
    if ((statusDispatcherThreads[consId] = pthread_mutex_lock (&accessCR)) != 0)
    {errno = statusDispatcherThreads[consId];
        perror ("error on entering monitor(CF)");
        statusDispatcherThreads[consId] = EXIT_FAILURE;
        pthread_exit (&statusDispatcherThreads[consId]);
    }

    pthread_once (&init, initialization);

    while ((ii_matrix == ri_matrix) && !fullMatrixMem)
    {
        /** Wait if FIFO Empty  */
        if ((statusDispatcherThreads[consId] = pthread_cond_wait (&fifoMatrixEmpty, &accessCR)) != 0)
        { errno = statusDispatcherThreads[consId];
            perror ("error on waiting in fifoEmpty");
            statusDispatcherThreads[consId] = EXIT_FAILURE;
            pthread_exit (&statusDispatcherThreads[consId]);
        }
        
    }

    *matrix = matrix_mem[ri_matrix];
    ri_matrix = (ri_matrix + 1) % K;

    fullMatrixMem = false;

    /** Let Thread Send Matrices & Receive Partial Results know that a Matrix has been retrieved */
    if ((statusDispatcherThreads[consId] = pthread_cond_signal (&fifoMatrixFull)) != 0)
    { errno = statusDispatcherThreads[consId];
        perror ("error on signaling in fifoFull");
        statusDispatcherThreads[consId] = EXIT_FAILURE;
        pthread_exit (&statusDispatcherThreads[consId]);
    }

    /** Exit monitor */
    if ((statusDispatcherThreads[consId] = pthread_mutex_unlock (&accessCR)) != 0)
    { errno = statusDispatcherThreads[consId];
        perror ("error on exiting monitor(CF)");
        statusDispatcherThreads[consId] = EXIT_FAILURE;
        pthread_exit (&statusDispatcherThreads[consId]);
    }

    /** There is Work to do*/
    return 0;
}




