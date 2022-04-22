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

int matrixToFIFO;
static unsigned int worker_waiting;
int matrixInfifo;

/** \brief Matrices storage region */
static struct Matrix matrix_mem[K];

/** \brief Files storage region */
static struct File_matrices file_mem[N];

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

    ii_matrix = ri_matrix = worker_waiting = 0;
    full_matrix_mem = false;

    /* initialize of synchronization points */
  pthread_cond_init (&fifoMatrixFull, NULL);
  pthread_cond_init (&fifoMatrixEmpty, NULL);
}

/**
 *  \brief Store a File (File_matrices) value in the data transfer region.
 *
 *  Operation carried out by the Main.
 *
 *  \param val File (File_matrices) to be stored
 */
void putFileInfo(struct File_matrices file_info){
    file_mem[ii_fileInfo] = file_info;
    ii_fileInfo = ii_fileInfo + 1;
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

    if(pthread_mutex_lock (&accessCR)!=0){
        printf("Main: error on entering monitor(CF)");
    }

    pthread_once (&init, initialization);

    while (full_matrix_mem){
        if(pthread_cond_wait (&fifoMatrixFull, &accessCR)!=0){
            printf("Main: error on waiting in fifoFull");
        } //TODO: VER se é necessarily.
    };

    matrix_mem[ii_matrix]= matrix;
    ii_matrix= (ii_matrix+1)%K;

    full_matrix_mem = (ii_matrix == ri_matrix);
    matrixToFIFO++;
    matrixInfifo++;
    if(pthread_cond_signal (&fifoMatrixEmpty)!=0){
        printf("Main: error on signaling in fifoEmpty");
    }

    if(pthread_mutex_unlock (&accessCR)!=0){
        printf("Main: error on exiting monitor(CF)");
    }

}

//TODO: ISTO NÃO ESTÁ A DAR NADA BEM. SÓ QUASE UM WORKER TRABALHA.
bool canigo(unsigned int consId){

    bool value;

    if ((statusWorks[consId] = pthread_mutex_lock (&accessCR)) != 0)
    {
        errno = statusWorks[consId];
        perror ("error on entering monitor(CF)");
        statusWorks[consId] = EXIT_FAILURE;
        pthread_exit (&statusWorks[consId]);
    }

    pthread_once (&init, initialization);

    if(matrixProcessed<128){
        if(matrixInfifo>worker_waiting){
            printf("Thread %u a VAI entrar com %u matrix to fifo e %u no fifo with %u waiting\n",consId,matrixToFIFO,matrixInfifo,worker_waiting);
            value=true;
        }
    }
    else{
        printf("Thread %u a VAI entrar com %u matrix to fifo e %u no fifo with %u waiting\n",consId,matrixToFIFO,matrixInfifo,worker_waiting);
        value=false;
    }

    /*
    if(matrixProcessed<128-worker_waiting  && ((matrixToFIFO-matrixInfifo))==0){
        printf("Thread %u a VAI entrar com %u matrix to fifo e %u no fifo with %u waiting\n",consId,matrixToFIFO,matrixInfifo,worker_waiting);
        value=true;
    }
    //Not all processed
    else if(matrixProcessed<128-worker_waiting && ((matrixToFIFO-matrixInfifo)>1)){
        printf("Thread %u a VAI entrar com %u matrix to fifo e %u no fifo with %u waiting\n",consId,matrixToFIFO,matrixInfifo,worker_waiting);
        value=true;
    }
    else{
        value=false;
        printf("Thread %u a NÃO vai entrar com %u matrix to fifo e %u no fifo with %u waiting\n",consId,matrixToFIFO,matrixInfifo,worker_waiting);

    }*/

    if ((statusWorks[consId] = pthread_mutex_unlock (&accessCR)) != 0)
    { errno = statusWorks[consId];
        perror ("error on exiting monitor(CF)");
        statusWorks[consId] = EXIT_FAILURE;
        pthread_exit (&statusWorks[consId]);
    }
    return value;
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
    struct Matrix val;

    printf("Worker %u : Aqui estou Manuel Acacio.\n",consId);

    if ((statusWorks[consId] = pthread_mutex_lock (&accessCR)) != 0)
    {errno = statusWorks[consId];
        perror ("error on entering monitor(CF)");
        statusWorks[consId] = EXIT_FAILURE;
        pthread_exit (&statusWorks[consId]);
    }

    pthread_once (&init, initialization);

    while ((ii_matrix == ri_matrix) && !full_matrix_mem)
    {
        printf("Worker %u : Aqui estou Esperando.\n",consId);

        if((matrixProcessed==128) || (worker_waiting>matrixInfifo && (128-matrixToFIFO<=worker_waiting))){

            printf("Thread %u a VAI SAIR com %u matrix to fifo e %u no fifo with %u waiting\n",consId,matrixToFIFO,matrixInfifo,worker_waiting);

            if ((statusWorks[consId] = pthread_mutex_unlock (&accessCR)) != 0)
            { errno = statusWorks[consId];
                perror ("error on exiting monitor(CF)");
                statusWorks[consId] = EXIT_FAILURE;
                pthread_exit (&statusWorks[consId]);
            }

            return 1;
        }

        worker_waiting=worker_waiting+1;

        printf("Thread %u a VAI esperar com %u matrix to fifo e %u no fifo with %u waiting\n",consId,matrixToFIFO,matrixInfifo,worker_waiting);

        if ((statusWorks[consId] = pthread_cond_wait (&fifoMatrixEmpty, &accessCR)) != 0)
        { errno = statusWorks[consId];
            perror ("error on waiting in fifoEmpty");
            statusWorks[consId] = EXIT_FAILURE;
            pthread_exit (&statusWorks[consId]);
        }
        worker_waiting--;
    }


    printf("Worker %u : VAI ENTRAR com %u matrix to fifo e  %u no fifo ficando %u waiting\n",consId,matrixToFIFO,matrixInfifo,worker_waiting);



    *matrix = matrix_mem[ri_matrix];

    ri_matrix = (ri_matrix + 1) % K;

    full_matrix_mem = false;

    matrixProcessed++;

    matrixInfifo--;



    if ((statusWorks[consId] = pthread_mutex_unlock (&accessCR)) != 0)
    { errno = statusWorks[consId];
        perror ("error on exiting monitor(CF)");
        statusWorks[consId] = EXIT_FAILURE;
        pthread_exit (&statusWorks[consId]);
    }

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

/**
 * Print in the terminal the results stored in the Shared Region
 * \param filesToProcess Number of Files
 */
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


