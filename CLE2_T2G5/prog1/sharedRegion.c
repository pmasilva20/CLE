/**
 *  \file fifo.c
 *
 *  \brief Assignment 2 : Problem 1 - Number of Words, Number of Words starting with a Vowel and Number of Words ending with a Consonant
 *
 *  Shared Region
 *
 *  Dispatcher Operations:
 *      \li putChunkText
 *      \li freeChunks
 *
 *  Workers Operations:
 *      \li getChunks
 *
 *  \author João Soares (93078) e Pedro Silva (93011)
 */

#include <stdbool.h>
#include <pthread.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include "structures.h"
#include "probConst.h"


/** \brief producer threads return status array */
extern int statusProd = 0;

/** \brief consumer threads return status array */
extern int *statusDispatcherThreads;

/** \brief Chunks storage region */
static struct ChunkText chunk_mem[K];

/** \brief flag signaling the data transfer region Chunk is full */
static bool full_text_chunk;

/** \brief Number of File Results Stored in Shared Region */
static int fileTextCount;

/** \brief insertion pointer for chunk_mem */
static unsigned int ii_chunk;

/** \brief retrieval pointer for chunk_mem */
static unsigned int  ri_chunk;


/** \brief consumers synchronization point when the Chunk data transfer region is empty */
static pthread_cond_t fifoChunkEmpty;

/** \brief producers synchronization point when the Chunk data transfer region is full */
static pthread_cond_t fifoChunkFull;

/** \brief consumers synchronization point when the Chunk data transfer region has no chunks in yet */
static pthread_cond_t fifoChunksPut;

/** \brief flag which signals that the main thread has finished processing all chunks of all files */
static bool finishedProcessing;

/** \brief flag which warrants that the data transfer region is initialized exactly once */
static pthread_once_t init = PTHREAD_ONCE_INIT;

/** \brief locking flag which warrants mutual exclusion inside the monitor */
static pthread_mutex_t accessCR = PTHREAD_MUTEX_INITIALIZER;

/** \brief Number of Text Chunks in Shared Region */
static int chunkCount;

/**
 *  \brief Initialization of the data transfer region.
 *
 *  Internal monitor operation.
 */
static void initialization (void)
{
    full_text_chunk = false;
    ii_chunk = 0;
    ri_chunk = 0;
    chunkCount = 0;
    fileTextCount = 0;
    finishedProcessing = false;

    pthread_cond_init (&fifoChunksPut, NULL);
    pthread_cond_init (&fifoChunkEmpty, NULL);
    pthread_cond_init (&fifoChunkFull, NULL);
}

/**
 * \brief Check if there are any chunks in Shared Region to process or if main is still processing new chunks.
 * If there are no chunks in Shared Region but main is still processing them, then it waits until a chunks is put.
 * Else retrieves a stored Text Chunk to be processed.
 *
 * Operation carried out by the workers.
 * @return True if there are chunks to be processed still
 */
bool getChunks(struct ChunkText* chunk, unsigned int consId) {

    if ((statusDispatcherThreads[consId] = pthread_mutex_lock (&accessCR)) != 0)                                   /* enter monitor */
    {
        errno = statusDispatcherThreads[consId];                                                            /* save error in errno */
        perror ("error on entering monitor(CF)");
        statusDispatcherThreads[consId] = EXIT_FAILURE;
        pthread_exit (statusDispatcherThreads[consId]);
    }

    pthread_once (&init, initialization);


    while((ii_chunk == ri_chunk) && !full_text_chunk){

        if(chunkCount == 0 && finishedProcessing){
            /** Exit monitor */
            if ((statusDispatcherThreads[consId] = pthread_mutex_unlock (&accessCR)) != 0)                                   /* enter monitor */
            {
                errno = statusDispatcherThreads[consId];                                                            /* save error in errno */
                perror ("error on exiting monitor(CF)");
                statusDispatcherThreads[consId] = EXIT_FAILURE;
                pthread_exit (statusDispatcherThreads[consId]);
            }
            return false;
        }

        if ((statusDispatcherThreads[consId] = pthread_cond_wait (&fifoChunkEmpty, &accessCR)) != 0)
        { errno = statusDispatcherThreads[consId];
            perror ("error on waiting in fifoChunkEmpty");
            statusDispatcherThreads[consId] = EXIT_FAILURE;
            pthread_exit (&statusDispatcherThreads[consId]);
        }

    }

    chunkCount--;
    (*chunk) = chunk_mem[ri_chunk];
    ri_chunk = (ri_chunk + 1) % K;
    full_text_chunk = false;

    /** Let Main know that a Matrix has been retrieved */
    if ((statusDispatcherThreads[consId] =  pthread_cond_signal (&fifoChunkFull)) != 0)
    { errno = statusDispatcherThreads[consId];
        perror ("error on signaling in fifoFull");
        statusDispatcherThreads[consId] = EXIT_FAILURE;
        pthread_exit (&statusDispatcherThreads[consId]);
    }

    /** Exit monitor */
    if ((statusDispatcherThreads[consId] = pthread_mutex_unlock (&accessCR)) != 0)
    { errno = statusDispatcherThreads[consId];
        perror ("error on exiting monitor(CF)");
        statusDispatcherThreads[consId]= EXIT_FAILURE;
        pthread_exit (&statusDispatcherThreads[consId]);
    }

    return true;
}


/**
 * \brief Insert a Text Chunk into Shared Region
 *
 * Operation done by dispatcher
 * @param chunk Chunk to be inserted
 */
void putChunkText(struct ChunkText chunk){

    /** Enter Monitor */
    if(pthread_mutex_lock (&accessCR)!=0){
        printf("Main: error on entering monitor(CF)");
    }

    pthread_once (&init, initialization);

    while (full_text_chunk){
        if ((pthread_cond_wait (&fifoChunkFull, &accessCR)) != 0)
        {
            printf("Main: error on waiting for Fifo Chunk Full");
        }
    };
    chunkCount++;
    /** Cleanup any remaining memory */
    if(chunk_mem[ii_chunk].chunk != NULL){
        free(chunk_mem[ii_chunk].chunk);
        chunk_mem[ii_chunk].chunk = NULL;
        chunk_mem[ii_chunk].filename = NULL;
        chunk_mem[ii_chunk].count = 0;
        chunk_mem[ii_chunk].fileId = 0;
    }


    chunk_mem[ii_chunk]= chunk;

    ii_chunk= (ii_chunk+1)%K;

    full_text_chunk = (ii_chunk == ri_chunk);

    if(pthread_cond_signal(&fifoChunksPut) != 0){
        printf("Main: error on signaling fifo chunks put");
    }
    if(pthread_cond_signal (&fifoChunkEmpty) != 0){
        printf("Main: error on signaling fifo chunk empty");
    }

    /** Exit Monitor */
    if(pthread_mutex_unlock (&accessCR)!=0){
        printf("Main: error on exiting monitor(CF)");
    }
}


/**
 * \brief Free any memory allocated previously in Shared Region.
 *
 * Operation carried out by dispatcher.
 */
void freeChunks(){
    /** Enter Monitor */
    if(pthread_mutex_lock (&accessCR)!=0){
        printf("Main: error on entering monitor(CF)");
    }

    pthread_once (&init, initialization);

    for(int i = 0; i < K; i++){
        if(chunk_mem[i].chunk != NULL){
            free(chunk_mem[i].chunk);
            chunk_mem[i].chunk = NULL;
        }
    }

    /** Exit Monitor */
    if(pthread_mutex_unlock (&accessCR)!=0){
        printf("Main: error on exiting monitor(CF)");
    }
}


