//
// Created by pmasilva20 on 19-04-2022.
//
#include <stdbool.h>
#include <pthread.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include "structures.h"




/** \brief producer threads return status array */
extern int statusProd = 0;

/** \brief Chunks storage region */
static struct Chunk_text chunk_mem[100];

/** \brief flag signaling the data transfer region Chunk is full */
static bool full_matrix_chunk;

/** \brief insertion pointer for chunk_mem */
static unsigned int ii_chunk;

/** \brief retrieval pointer for chunk_mem */
static unsigned int  ri_chunk;

/** \brief consumers synchronization point when the data transfer region is empty */
static pthread_cond_t fifoChunkEmpty;

static pthread_cond_t fifoChunkFull;

/** \brief flag which warrants that the data transfer region is initialized exactly once */
static pthread_once_t init = PTHREAD_ONCE_INIT;

/** \brief locking flag which warrants mutual exclusion inside the monitor */
static pthread_mutex_t accessCR = PTHREAD_MUTEX_INITIALIZER;

static void initialization (void)
{
    full_matrix_chunk = false;
    ii_chunk = 0;
    ri_chunk = 0;
    pthread_cond_init (&fifoChunkEmpty, NULL);
    pthread_cond_init (&fifoChunkFull, NULL);
}


struct Chunk_text getChunkText(){
    struct Chunk_text chunk;
    //Enter monitor
    pthread_mutex_lock (&accessCR);

    pthread_once (&init, initialization);

    while((ii_chunk == ri_chunk) && !full_matrix_chunk){
        pthread_cond_wait (&fifoChunkEmpty, &accessCR);
    }

    chunk = chunk_mem[ri_chunk];
    ri_chunk = (ri_chunk + 1) % 100;
    full_matrix_chunk = false;

    pthread_cond_signal (&fifoChunkFull);
    pthread_mutex_unlock (&accessCR);

    return chunk;
}

void putChunkText(struct Chunk_text chunk){

    //Check if I can enter
    if ((statusProd = pthread_mutex_lock (&accessCR)) != 0)                                   /* enter monitor */
    {
        errno = statusProd;                                                            /* save error in errno */
        perror ("error on entering monitor(CF)");
        statusProd = EXIT_FAILURE;
        pthread_exit (statusProd);
    }

    //Init only once
    pthread_once (&init, initialization);

    while (full_matrix_chunk){
        if ((statusProd = pthread_cond_wait (&fifoChunkFull, &accessCR)) != 0)
        {
            errno = statusProd;                                                          /* save error in errno */
            perror ("error on waiting in fifoFull");
            statusProd = EXIT_FAILURE;
            pthread_exit (&statusProd);
        }
    };
    chunk_mem[ii_chunk]= chunk;

    ii_chunk= (ii_chunk+1)%100;

    full_matrix_chunk = (ii_chunk == ri_chunk);

    pthread_cond_signal (&fifoChunkEmpty);
    pthread_mutex_unlock (&accessCR);
}
