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

/** Matrices storage region */
static struct Matrix matrix_mem[128]; //TODO: Perguntar sobre o tamanho disto ao prof

static struct File_matrices file_mem[10];


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

/**
 *  \brief Initialization of the data transfer region.
 *
 *  Internal monitor operation.
 */

static void initialization (void)
{

  ii_matrix = ri_matrix = 0;
    full_matrix_mem = false;

    /* initialize of synchronization points */
  pthread_cond_init (&fifoMatrixFull, NULL);
  pthread_cond_init (&fifoMatrixEmpty, NULL);
}


void putFileInfo(struct File_matrices file_info){
    file_mem[ii_fileInfo] = file_info;
    ii_fileInfo = ii_fileInfo + 1;
}

// TODO: Meter aqui monitor por causa de K e tamanho dos arrays (Está a funcionar)!
void putMatrixVal(struct Matrix matrix){
    while (full_matrix_mem){
    };
    matrix_mem[ii_matrix]= matrix;

    ii_matrix= (ii_matrix+1)%K;

    full_matrix_mem = (ii_matrix == ri_matrix);

    pthread_cond_signal (&fifoMatrixEmpty);
}

struct Matrix getMatrixVal(unsigned int consId)
{
    struct Matrix val;
    /* retrieved value */
    if ((statusWorks[consId] = pthread_mutex_lock (&accessCR)) != 0)
    { errno = statusWorks[consId];
        perror ("error on entering monitor(CF)");
        statusWorks[consId] = EXIT_FAILURE;
        pthread_exit (&statusWorks[consId]);
    }

    pthread_once (&init, initialization);


    while ((ii_matrix == ri_matrix) && !full_matrix_mem)
    { if ((statusWorks[consId] = pthread_cond_wait (&fifoMatrixEmpty, &accessCR)) != 0)
        { errno = statusWorks[consId];
            perror ("error on waiting in fifoEmpty");
            statusWorks[consId] = EXIT_FAILURE;
            pthread_exit (&statusWorks[consId]);
        }
    }

    val = matrix_mem[ri_matrix];
    ri_matrix = (ri_matrix + 1) % K;
    full_matrix_mem = false;
    matrixProcessed++;

    if ((statusWorks[consId] = pthread_mutex_unlock (&accessCR)) != 0)
    { errno = statusWorks[consId];
        perror ("error on exiting monitor(CF)");
        statusWorks[consId] = EXIT_FAILURE;
        pthread_exit (&statusWorks[consId]);
    }

    return val;
}
//TODO: Verificar Monitor aqui!
void putResults(struct Matrix_result result,unsigned int consId){

    if ((statusWorks[consId] = pthread_mutex_lock (&accessCR)) != 0)
    { errno = statusWorks[consId];
        perror ("error on entering monitor(CF)");
        statusWorks[consId] = EXIT_FAILURE;
        pthread_exit (&statusWorks[consId]);
    }

    pthread_once (&init, initialization);

    size_t arraySize = sizeof(file_mem) / sizeof(*file_mem);
    for (int x = 0; x < arraySize; x++){
        if(file_mem[x].id==result.fileid){
            int id =result.id;
            file_mem[x].determinant_result[id]=result;
            break;
        }
    }

    if ((statusWorks[consId] = pthread_mutex_unlock (&accessCR)) != 0)
    { errno = statusWorks[consId];
        perror ("error on exiting monitor(CF)");
        statusWorks[consId] = EXIT_FAILURE;
        pthread_exit (&statusWorks[consId]);
    }
}
//TODO: Alterar dependo de quando fazer o print.
void getResults(int filesToProcess){
    for (int x = 0; x < filesToProcess; x++){
        printf("File: %s\n",file_mem[x].name);
        for (int a = 0; a < file_mem[x].numberOfMatrices; a++){
            printf("Matrix %d :\n",file_mem[x].determinant_result[a].id+1);
            printf("The determinant is %.3e\n",file_mem[x].determinant_result[a].determinant);
        }
        printf("\n");
    }
}


