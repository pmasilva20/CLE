/**
 *  \file fifo.c
 *
 *  \brief Assignment 1 : Problem 1 - Number of Words, Number of Words starting with a Vowel and Number of Words ending with a Consonant
 *
 *  Shared Region
 *
 *  Main Operations:
 *      \li finishedProcessingChunks
 *      \li getFileText
 *      \li putChunkText
 *
 *  Workers Operations:
 *      \li getChunks
 *      \li putFileText
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
extern int *statusWorks;

/** \brief Chunks storage region */
static struct ChunkText chunk_mem[K];

/** \brief Files Text storage region */
static struct FileText files_mem[N];

/** \brief flag signaling the data transfer region File Text is full */
static bool full_file_text;

/** \brief flag signaling the data transfer region Chunk is full */
static bool full_text_chunk;

/** \brief Number of File Results Stored in Shared Region */
static int fileTextCount;

/** \brief insertion pointer for chunk_mem */
static unsigned int ii_chunk;

/** \brief retrieval pointer for chunk_mem */
static unsigned int  ri_chunk;

/** \brief insertion pointer for files_mem */
static unsigned int ii_file;

/** \brief retrieval pointer for files_mem */
static unsigned int  ri_file;

/** \brief consumers synchronization point when the Chunk data transfer region is empty */
static pthread_cond_t fifoChunkEmpty;

/** \brief producers synchronization point when the Chunk data transfer region is full */
static pthread_cond_t fifoChunkFull;

/** \brief consumers synchronization point when the File data transfer region is empty */
static pthread_cond_t fifoFileEmpty;

/** \brief producers synchronization point when the File data transfer region is full */
static pthread_cond_t fifoFileFull;

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
    full_file_text = false;
    finishedProcessing = false;
    ii_file = 0;
    ri_file = 0;

    pthread_cond_init (&fifoChunksPut, NULL);
    pthread_cond_init (&fifoFileEmpty, NULL);
    pthread_cond_init (&fifoFileFull, NULL);
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

    if ((statusWorks[consId] = pthread_mutex_lock (&accessCR)) != 0)                                   /* enter monitor */
    {
        errno = statusWorks[consId];                                                            /* save error in errno */
        perror ("error on entering monitor(CF)");
        statusWorks[consId] = EXIT_FAILURE;
        pthread_exit (statusWorks[consId]);
    }

    pthread_once (&init, initialization);


    while((ii_chunk == ri_chunk) && !full_text_chunk){

        if(chunkCount == 0 && finishedProcessing){
            /** Exit monitor */
            if ((statusWorks[consId] = pthread_mutex_unlock (&accessCR)) != 0)                                   /* enter monitor */
            {
                errno = statusWorks[consId];                                                            /* save error in errno */
                perror ("error on exiting monitor(CF)");
                statusWorks[consId] = EXIT_FAILURE;
                pthread_exit (statusWorks[consId]);
            }
            return false;
        }

        if ((statusWorks[consId] = pthread_cond_wait (&fifoChunkEmpty, &accessCR)) != 0)
        { errno = statusWorks[consId];
            perror ("error on waiting in fifoChunkEmpty");
            statusWorks[consId] = EXIT_FAILURE;
            pthread_exit (&statusWorks[consId]);
        }

    }

    chunkCount--;
    (*chunk) = chunk_mem[ri_chunk];
    ri_chunk = (ri_chunk + 1) % K;
    full_text_chunk = false;

    /** Let Main know that a Matrix has been retrieved */
    if ((statusWorks[consId] =  pthread_cond_signal (&fifoChunkFull)) != 0)
    { errno = statusWorks[consId];
        perror ("error on signaling in fifoFull");
        statusWorks[consId] = EXIT_FAILURE;
        pthread_exit (&statusWorks[consId]);
    }

    /** Exit monitor */
    if ((statusWorks[consId] = pthread_mutex_unlock (&accessCR)) != 0)
    { errno = statusWorks[consId];
        perror ("error on exiting monitor(CF)");
        statusWorks[consId]= EXIT_FAILURE;
        pthread_exit (&statusWorks[consId]);
    }

    return true;
}
/**
 * \brief Signal that all text files have been read by main and divided into chunks.
 * Signals any awaiting worker thread.
 *
 * Operation carried out by main.
 */
void finishedProcessingChunks(){
    /** Enter Monitor */
    if(pthread_mutex_lock (&accessCR)!=0){
        printf("Main:Error on entering monitor(CF)");
    }

    pthread_once (&init, initialization);
    finishedProcessing = true;
    pthread_cond_broadcast(&fifoChunkEmpty);

    /** Exit monitor */
    if(pthread_mutex_unlock (&accessCR)!=0){
        printf("Main: error on exiting monitor(CF)");
    }
}

/**
 * \brief Retrieve a stored File information after all chunks have been processed.
 *
 * Operation carried out by main.
 * @param fileId File Id of the file that is to be retrieved
 * @return FileText structure with file information
 */
struct FileText* getFileText(int fileId){

    /** Enter Monitor */
    if(pthread_mutex_lock (&accessCR)!=0){
        printf("Main: error on entering monitor(CF)");
    }

    pthread_once (&init, initialization);

    for(int i = 0; i < fileTextCount; i++){
        if(files_mem[i].fileId == fileId){
            pthread_mutex_unlock (&accessCR);
            return &files_mem[i];
        }
    }

    /** Exit Monitor */
    if(pthread_mutex_unlock (&accessCR)!=0){
        printf("Main: error on exiting monitor(CF)");
    }

    return NULL;
}

/**
 * \brief Store information relating to the processing of a Text Chunk in the Shared Region.
 * If information for this file has not been stored in Shared Region already, then a new structure is allocated.
 * If information for this file has been stored in Shared Region already, then chunk information is added.
 * Operation is carried out by the workers.
 *
 * @param consId Worker ID
 * @param nWords Number of words in chunk
 * @param nVowelStartWords Number of words starting with a vowel in chunk
 * @param nConsonantEndWord Number of words ending with a consonant in chunk
 * @param fileID File Id that identifies file corresponding to chunk
 * @param filename File name of file that corresponds to chunk
 */
void putFileText(int nWords, int nVowelStartWords, int nConsonantEndWord, int fileID, char *filename, unsigned int consId) {

    /** Check if I can enter monitor*/
    if ((statusWorks[consId] = pthread_mutex_lock (&accessCR)) != 0)                                   /* enter monitor */
    {
        errno = statusWorks[consId];                                                            /* save error in errno */
        perror ("error on entering monitor(CF)");
        statusWorks[consId] = EXIT_FAILURE;
        pthread_exit (statusWorks[consId]);
    }

    pthread_once (&init, initialization);

    while (full_file_text){
        if ((statusWorks[consId] = pthread_cond_wait (&fifoFileFull, &accessCR)) != 0)
        {
            errno = statusWorks[consId];                                                          /* save error in errno */
            perror ("error on waiting in fifoFull");
            statusWorks[consId] = EXIT_FAILURE;
            pthread_exit (&statusWorks[consId]);
        }
    };

    struct FileText* text;
    bool foundText = false;
    /** Check of this chunk corresponding file already has information stored in Shared Region */
    for(int i = 0; i < fileTextCount; i++){
        if(files_mem[i].fileId == fileID){
            text = &files_mem[i];
            foundText = true;
            break;
        }
    }

    /** Previous file statistics have already been stored, update them */
    if(foundText){
        (*text).nConsonantEndWord += nConsonantEndWord;
        (*text).nVowelStartWords += nVowelStartWords;
        (*text).nWords += nWords;
        (*text).fileId = fileID;
        (*text).name = filename;
    }
    /** No previous file statistics, make a new structure and store what we have */
    else{
        struct FileText newText;
        newText.nConsonantEndWord = nConsonantEndWord;
        newText.nVowelStartWords = nVowelStartWords;
        newText.nWords = nWords;
        newText.fileId = fileID;
        newText.name = filename;

        fileTextCount++;
        files_mem[ii_file]= newText;

        ii_file= (ii_file+1)%N;

        full_file_text = (ii_file == ri_file);
    }



    pthread_cond_signal (&fifoFileEmpty);

    /** Exit monitor */
    if ((statusWorks[consId] = pthread_mutex_unlock (&accessCR)) != 0)
    { errno = statusWorks[consId];
        perror ("error on exiting monitor(CF)");
        statusWorks[consId]= EXIT_FAILURE;
        pthread_exit (&statusWorks[consId]);
    }
}

/**
 * \brief Insert a Text Chunk into Shared Region
 *
 * Operation done by main
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
 * Operation carried out by main.
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


