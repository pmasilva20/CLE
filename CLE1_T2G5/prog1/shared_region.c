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
 *      \li hasChunksLeft
 *      \li putFileText
 *      \li getChunkText
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

/** \brief Chunks storage region */
static struct Chunk_text chunk_mem[K];

/** \brief Files Text storage region */
static struct File_text files_mem[N];

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
 *
 * Operation carried out by the workers.
 * @return True if there are chunks to be processed still
 */
bool hasChunksLeft(){

    pthread_mutex_lock (&accessCR);

    pthread_once (&init, initialization);

    if(!finishedProcessing && chunkCount == 0){
        pthread_cond_wait (&fifoChunksPut, &accessCR);
    }

    pthread_mutex_unlock (&accessCR);

    return !finishedProcessing || chunkCount > 0;
}
/**
 * \brief Signal that all text files have been read by main and divided into chunks.
 * Signals any awaiting worker thread.
 *
 * Operation carried out by main.
 */
void finishedProcessingChunks(){
    pthread_mutex_lock (&accessCR);

    pthread_once (&init, initialization);
    finishedProcessing = true;
    pthread_cond_broadcast(&fifoChunksPut);

    pthread_mutex_unlock (&accessCR);
}

/**
 * \brief Retrieve a stored File information after all chunks have been processed.
 *
 * Operation carried out by main.
 * @param fileId File Id of the file that is to be retrieved
 * @return File_text structure with file information
 */
struct File_text* getFileText(int fileId){

    if ((statusProd = pthread_mutex_lock (&accessCR)) != 0)                                   /* enter monitor */
    {
        errno = statusProd;                                                                         /* save error in errno */
        perror ("error on entering monitor(CF)");
        statusProd = EXIT_FAILURE;
        pthread_exit (statusProd);
    }

    pthread_once (&init, initialization);

    for(int i = 0; i < fileTextCount; i++){
        if(files_mem[i].fileId == fileId){
            pthread_mutex_unlock (&accessCR);
            return &files_mem[i];
        }
    }

    pthread_mutex_unlock (&accessCR);

    return NULL;
}

/**
 * \brief Store information relating to the processing of a Text Chunk in the Shared Region.
 * If information for this file has not been stored in Shared Region already, then a new structure is allocated.
 * If information for this file has been stored in Shared Region already, then chunk information is added.
 * Operation is carried out by the workers.
 *
 * @param nWords Number of words in chunk
 * @param nVowelStartWords Number of words starting with a vowel in chunk
 * @param nConsonantEndWord Number of words ending with a consonant in chunk
 * @param fileID File Id that identifies file corresponding to chunk
 * @param filename File name of file that corresponds to chunk
 */
void putFileText(int nWords, int nVowelStartWords, int nConsonantEndWord, int fileID, char* filename){

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

    while (full_file_text){
        if ((statusProd = pthread_cond_wait (&fifoFileFull, &accessCR)) != 0)
        {
            errno = statusProd;                                                          /* save error in errno */
            perror ("error on waiting in fifoFull");
            statusProd = EXIT_FAILURE;
            pthread_exit (&statusProd);
        }
    };

    struct File_text* text;
    bool foundText = false;
    int idxFound = -1;
    for(int i = 0; i < fileTextCount; i++){
        if(files_mem[i].fileId == fileID){
            text = &files_mem[i];
            idxFound = i;
            foundText = true;
            break;
        }
    }

    if(foundText){
        (*text).nConsonantEndWord += nConsonantEndWord;
        (*text).nVowelStartWords += nVowelStartWords;
        (*text).nWords += nWords;
        (*text).fileId = fileID;
        (*text).name = filename;
    }
    else{
        struct File_text newText;
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
    pthread_mutex_unlock (&accessCR);
}

/**
 * \brief Retrieve a stored Text Chunk to be processed.
 *
 * Operation carried out by the workers.
 * @param fileId File Id of the file that is to be retrieved
 * @return File_text structure with file information
 */
struct Chunk_text* getChunkText(){
    struct Chunk_text* chunk;
    //Enter monitor
    pthread_mutex_lock (&accessCR);

    pthread_once (&init, initialization);

    if(chunkCount == 0){
        pthread_mutex_unlock (&accessCR);
        return NULL;
    }

    while((ii_chunk == ri_chunk) && !full_text_chunk){
        pthread_cond_wait (&fifoChunkEmpty, &accessCR);
    }

    chunkCount--;
    chunk = &chunk_mem[ri_chunk];
    ri_chunk = (ri_chunk + 1) % K;
    full_text_chunk = false;

    pthread_cond_signal (&fifoChunkFull);
    pthread_mutex_unlock (&accessCR);

    return chunk;
}

/**
 * \brief Insert a Text Chunk into Shared Region
 * @param chunk Chunk to be inserted
 */
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

    while (full_text_chunk){
        if ((statusProd = pthread_cond_wait (&fifoChunkFull, &accessCR)) != 0)
        {
            errno = statusProd;                                                          /* save error in errno */
            perror ("error on waiting in fifoFull");
            statusProd = EXIT_FAILURE;
            pthread_exit (&statusProd);
        }
    };
    chunkCount++;
    chunk_mem[ii_chunk]= chunk;

    ii_chunk= (ii_chunk+1)%K;

    full_text_chunk = (ii_chunk == ri_chunk);

    pthread_cond_signal(&fifoChunksPut);
    pthread_cond_signal (&fifoChunkEmpty);
    pthread_mutex_unlock (&accessCR);
}


