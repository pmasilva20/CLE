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

/** \brief Files Text storage region */
static struct File_text files_mem[100];

/** \brief flag signaling the data transfer region File Text is full */
static bool full_file_text;

/** \brief flag signaling the data transfer region Chunk is full */
static bool full_text_chunk;

static int fileTextCount;

/** \brief insertion pointer for chunk_mem */
static unsigned int ii_chunk;

/** \brief retrieval pointer for chunk_mem */
static unsigned int  ri_chunk;

/** \brief insertion pointer for files_mem */
static unsigned int ii_file;

/** \brief retrieval pointer for files_mem */
static unsigned int  ri_file;

/** \brief consumers synchronization point when the data transfer region is empty */
static pthread_cond_t fifoChunkEmpty;

static pthread_cond_t fifoChunkFull;

/** \brief consumers synchronization point when the data transfer region is empty */
static pthread_cond_t fifoFileEmpty;

static pthread_cond_t fifoFileFull;

/** \brief flag which warrants that the data transfer region is initialized exactly once */
static pthread_once_t init = PTHREAD_ONCE_INIT;

/** \brief locking flag which warrants mutual exclusion inside the monitor */
static pthread_mutex_t accessCR = PTHREAD_MUTEX_INITIALIZER;

static int chunkCount;


static void initialization (void)
{
    full_text_chunk = false;
    ii_chunk = 0;
    ri_chunk = 0;
    chunkCount = 0;
    fileTextCount = 0;
    full_file_text = false;
    ii_file = 0;
    ri_file = 0;

    pthread_cond_init (&fifoFileEmpty, NULL);
    pthread_cond_init (&fifoFileFull, NULL);
    pthread_cond_init (&fifoChunkEmpty, NULL);
    pthread_cond_init (&fifoChunkFull, NULL);
}

int getChunkCount(){
    //Enter monitor
    pthread_mutex_lock (&accessCR);

    int count = chunkCount;
    pthread_mutex_unlock (&accessCR);
    return count;
}


struct File_text getFileText(int fileId){
    struct File_text file_text;
    //Enter monitor
    pthread_mutex_lock (&accessCR);

    pthread_once (&init, initialization);

    for(int i = 0; i < fileTextCount; i++){
        if(files_mem[i].fileId == fileId) return files_mem[i];
    }

    pthread_mutex_unlock (&accessCR);

    struct File_text newFile;
    return newFile;
}

void putFileText(int nWords, int nVowelStartWords, int nConsonantEndWord, int fileID){

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

    struct File_text text;
    bool foundText = false;
    int idxFound = -1;
    for(int i = 0; i < fileTextCount; i++){
        if(files_mem[i].fileId == fileID){
            text = files_mem[i];
            idxFound = i;
            foundText = true;
            break;
        }
    }
    if(foundText){
        text.nConsonantEndWord += nConsonantEndWord;
        text.nVowelStartWords += nVowelStartWords;
        text.nWords += nWords;
        text.fileId = fileID;
        files_mem[idxFound] = text;
    }
    else{
        text.nConsonantEndWord = nConsonantEndWord;
        text.nVowelStartWords = nVowelStartWords;
        text.nWords = nWords;
        text.fileId = fileID;

        fileTextCount++;
        files_mem[ii_file]= text;

        ii_file= (ii_file+1)%100;

        full_file_text = (ii_file == ri_file);
    }



    pthread_cond_signal (&fifoFileEmpty);
    pthread_mutex_unlock (&accessCR);
}


struct Chunk_text getChunkText(){
    struct Chunk_text chunk;
    //Enter monitor
    pthread_mutex_lock (&accessCR);

    pthread_once (&init, initialization);

    while((ii_chunk == ri_chunk) && !full_text_chunk){
        pthread_cond_wait (&fifoChunkEmpty, &accessCR);
    }

    chunkCount--;
    chunk = chunk_mem[ri_chunk];
    ri_chunk = (ri_chunk + 1) % 100;
    full_text_chunk = false;

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

    ii_chunk= (ii_chunk+1)%100;

    full_text_chunk = (ii_chunk == ri_chunk);

    pthread_cond_signal (&fifoChunkEmpty);
    pthread_mutex_unlock (&accessCR);
}
