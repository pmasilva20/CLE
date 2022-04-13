/**
 *  \file fifo.c (implementation file)
 *
 *  \brief Problem name: Producers / Consumers.
 *
 *  Synchronization based on monitors.
 *  Both threads and the monitor are implemented using the pthread library which enables the creation of a
 *  monitor of the Lampson / Redell type.
 *
 *  Data transfer region implemented as a monitor.
 *
 *  Definition of the operations carried out by the producers / consumers:
 *     \li putVal
 *     \li getVal.
 *
 *  \author António Rui Borges - March 2019
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

extern int matrixProcessed;

/** Matrices storage region */
static struct Matrix matrix_mem[256];

static struct File_matrices file_mem[256];


/** \brief insertion pointer for file_mem */
static unsigned int  ii_fileInfo;

/** \brief insertion pointer for matrix_mem */
static unsigned int  ii_matrix;

/** \brief insertion pointer for matrix_mem */
static unsigned int  ri_matrix;


/** \brief flag signaling the data transfer region Matrix is full */
static bool full_matrix_mem;

/** \brief producers synchronization point when the data transfer region is full */
static pthread_cond_t fifoMatrixFull;

/** \brief consumers synchronization point when the data transfer region is empty */
static pthread_cond_t fifoMatrixEmpty;

/** \brief locking flag which warrants mutual exclusion inside the monitor */
static pthread_mutex_t accessCR = PTHREAD_MUTEX_INITIALIZER;

/** \brief flag which warrants that the data transfer region is initialized exactly once */
static pthread_once_t init = PTHREAD_ONCE_INIT;;

/** \brief producers synchronization point when the data transfer region is full */
static pthread_cond_t fifoFull;

/** \brief consumers synchronization point when the data transfer region is empty */
static pthread_cond_t fifoEmpty;

/**
 *  \brief Initialization of the data transfer region.
 *
 *  Internal monitor operation.
 */

static void initialization (void)
{
                                                                                   /* initialize FIFO in empty state */
  ii_matrix = ri_matrix = 0;                                        /* FIFO insertion and retrieval pointers set to the same value */
    full_matrix_mem = false;                                                                                  /* FIFO is not full */

  pthread_cond_init (&fifoMatrixFull, NULL);                                 /* initialize producers synchronization point */
  pthread_cond_init (&fifoMatrixEmpty, NULL);                                /* initialize consumers synchronization point */
}

/**
 *  \brief Store a value in the data transfer region.
 *
 *  Operation carried out by the producers.
 *
 *  \param prodId producer identification
 *  \param val value to be stored
 */

void putFileInfo(struct File_matrices file_info){
    file_mem[ii_fileInfo] = file_info;                                                                          /* store value in the FIFO */
    ii_fileInfo = ii_fileInfo + 1;
}

void putMatrixVal(struct Matrix matrix){
    matrix_mem[ii_matrix]= matrix;
    //print_matrix_details(matrix_mem[ii_matrix]);
    ii_matrix= (ii_matrix+1)%K;
    full_matrix_mem = (ii_matrix == ri_matrix);
    //TODO: let a consumer know that a value has been stored
    pthread_cond_signal (&fifoMatrixEmpty);
}

struct Matrix getMatrixVal(unsigned int consId)
{
    struct Matrix val;
    /* retrieved value */
    if ((statusWorks[consId] = pthread_mutex_lock (&accessCR)) != 0)                                   /* enter monitor */
    { errno = statusWorks[consId];                                                            /* save error in errno */
        perror ("error on entering monitor(CF)");
        statusWorks[consId] = EXIT_FAILURE;
        pthread_exit (&statusWorks[consId]);
    }

    pthread_once (&init, initialization);                                              /* internal data initialization */

    //TODO: a dar erro aqui!
    while ((ii_matrix == ri_matrix) && !full_matrix_mem)                                           /* wait if the data transfer region is empty */
    { if ((statusWorks[consId] = pthread_cond_wait (&fifoMatrixEmpty, &accessCR)) != 0)
        { errno = statusWorks[consId];                                                          /* save error in errno */
            perror ("error on waiting in fifoEmpty");
            statusWorks[consId] = EXIT_FAILURE;
            pthread_exit (&statusWorks[consId]);
        }
    }

    val = matrix_mem[ri_matrix];                                                                   /* retrieve a  value from the FIFO */
    ri_matrix = (ri_matrix + 1) % K;
    full_matrix_mem = false;
    matrixProcessed++;
    if ((statusWorks[consId] = pthread_mutex_unlock (&accessCR)) != 0)                                   /* exit monitor */
    { errno = statusWorks[consId];                                                             /* save error in errno */
        perror ("error on exiting monitor(CF)");
        statusWorks[consId] = EXIT_FAILURE;
        pthread_exit (&statusWorks[consId]);
    }

    return val;
}
//TODO: Verificar Monitor aqui
void putResults(struct Matrix_result result,unsigned int consId){

    if ((statusWorks[consId] = pthread_mutex_lock (&accessCR)) != 0)                                   /* enter monitor */
    { errno = statusWorks[consId];                                                            /* save error in errno */
        perror ("error on entering monitor(CF)");
        statusWorks[consId] = EXIT_FAILURE;
        pthread_exit (&statusWorks[consId]);
    }

    pthread_once (&init, initialization);                                              /* internal data initialization */

    size_t arraySize = sizeof(file_mem) / sizeof(*file_mem);
    for (int x = 0; x < arraySize; x++){
        if(file_mem[x].id==result.fileid){
            int id =result.id;
            file_mem[x].determinant_result[id]=result;
            break;
        }
    }

    if ((statusWorks[consId] = pthread_mutex_unlock (&accessCR)) != 0)                                   /* exit monitor */
    { errno = statusWorks[consId];                                                             /* save error in errno */
        perror ("error on exiting monitor(CF)");
        statusWorks[consId] = EXIT_FAILURE;
        pthread_exit (&statusWorks[consId]);
    }
}
//TODO: Mallocs por causa dos sizes do array.
void getResults(){
    size_t arraySize = sizeof(file_mem) / sizeof(*file_mem);
    for (int x = 0; x < arraySize; x++){
        if(x==0){
            printf("File: %s\n",file_mem[x].name);
            for (int a = 0; a < 128; a++){
                printf("Fileid: %d\n",file_mem[x].determinant_result[a].fileid);
                printf("Matrixid: %d\n",file_mem[x].determinant_result[a].id+1);
                printf("The determinant is %.3e\n",file_mem[x].determinant_result[a].determinant);
            }
        }

        //size_t arraySize_r = sizeof(file_mem[x].determinant_result) / sizeof(*file_mem[x].determinant_result);
    }
}


