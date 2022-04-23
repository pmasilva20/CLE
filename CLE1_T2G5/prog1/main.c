/**
 *  \file main.c
 *
 *  \brief Assignment 1 : Problem 1 - Number of Words, Number of Words starting with a Vowel and Number of Words ending with a Consonant
 *
 *  Main Program
 *
 *  \author João Soares (93078) e Pedro Silva (93011)
 */

#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <libgen.h>
#include <unistd.h>
#include <pthread.h>
#include "structures.h"
#include "func.h"
#include "fifo.h"
#include "prob1_processing.h"
#include "worker.h"


/** \brief consumer threads return status array */
int *statusWorks;

static void printUsage (char *cmdName);
static void *worker (void *par);


int main (int argc, char** argv){

    /** time limits **/
    struct timespec start, finish;

    /** \brief Number of Worker Threads */
    int numberWorkers = 5;

    /** \brief  List of Files Names**/
    char* fileNames[argc];

    /** \brief File ID */
    int fileid=0;

    /** selected option */
    int opt;
    opterr = 0;
    /**
     * Command Line Processing
     */
    do {
        switch ((opt = getopt(argc, argv, ":f:t:h"))) {
            case 'f':
                if (optarg[0] == '-') {
                    fprintf(stderr, "%s: file name is missing\n", basename(argv[0]));
                    printUsage(basename(argv[0]));
                    return EXIT_FAILURE;
                }
                /** Copy Filename*/
                fileNames[fileid] = optarg;
                /** Update fileid */
                fileid++;

                break;
            case 't':
                numberWorkers = atoi(optarg);
                break;
            case 'h' : /* help mode */
                printUsage(basename(argv[0]));
                return EXIT_SUCCESS;
            case '?': /* invalid option */
                fprintf (stderr, "%s: invalid option\n", basename (argv[0]));
                printUsage (basename (argv[0]));
                return EXIT_FAILURE;
            case -1: break;
        }
    } while (opt != -1);

    if (argc == 1){
        fprintf (stderr, "%s: invalid format\n", basename (argv[0]));
        printUsage (basename (argv[0]));
        return EXIT_FAILURE;
    }


    /** Workers status array */
    statusWorks = malloc(sizeof(int)*numberWorkers);

    /** Workers internal thread id array */
    pthread_t tIdWorkers[numberWorkers];

    /** Workers application defined thread id array*/
    unsigned int works[numberWorkers];

    /** Pointer to execution status */
    int *status_p;

    for (int i = 0; i < numberWorkers; i++)
        works[i] = i;

    srandom ((unsigned int) getpid ());

    /** Begin of Time measurement */
    clock_gettime (CLOCK_MONOTONIC_RAW, &start);

    /**
     * Generation of intervening Workers threads
     */
    for (int i = 0; i < numberWorkers; i++) {

        if (pthread_create(&tIdWorkers[i], NULL, worker, &works[i]) !=0)
        {
            perror("error on creating thread worker");
            exit(EXIT_FAILURE);
        }
        else{
            printf("Thread Worker Created %d !\n", i);
        }
    }
    /**
     * For each file make text chunks taking into account variable chunk size
     * Store each chunk in Shared Region
     */
    for(int i = 0; i < fileid; i++){
        makeChunks(fileNames[i],i,10);
    }
    /**Signalize any waiting workers that all chunks have been made by main */
    finishedProcessingChunks();


    /** Waiting for the termination of the Workers threads */
    for(int i = 0; i < numberWorkers; i++){
        if (pthread_join (tIdWorkers[i], (void *) &status_p) != 0){
            perror ("error on waiting for thread producer");
            exit (EXIT_FAILURE);
        }
        printf ("thread worker, with id %u, has terminated\n", i);
    }
    /** End of measurement */
    clock_gettime (CLOCK_MONOTONIC_RAW, &finish);

    /** Get all Files statistics and print to console */
    for(int i = 0; i < fileid; i++){
        struct File_text* text = getFileText(i);
        if(text != NULL){
            printf("File name: %s\n",(*text).name);
            printf("Total number of words = %d\n",(*text).nWords);
            printf("N. of words beginning with a vowel = %d\n",(*text).nVowelStartWords);
            printf("N. of words ending with a consonant = %d\n",(*text).nConsonantEndWord);
            printf("\n");
        }
        else printf("Error retrieving files statistics for file %d",i);
    }

    /** Print Elapsed Time */
    printf ("\nElapsed time = %.6f s\n",  (finish.tv_sec - start.tv_sec) / 1.0 + (finish.tv_nsec - start.tv_nsec) / 1000000000.0);

}

/**
 * \brief Worker Function checks if there's any Text Chunks to process
 * @param par A pointer to application defined worker identification
 */
static void *worker (void *par)
{
    /** Worker ID */
    unsigned int id = *((unsigned int *) par);

    printf("Worker %d ready!\n",id);

    /** While there are any chunks to process*/
    while(hasChunksLeft()){
        /** Try to acquire a Text Chunk*/
        struct Chunk_text* chunk = getChunkText();
        /** Process Text Chunk and store results in Shared Region*/
        if(chunk != NULL)processChunk(*chunk);
        //printf("Worker:%d Remaining chunksToProcess %d\n",id,getChunkCount());
    }
    /** Exit with success after handling all chunks*/
    statusWorks[id] = EXIT_SUCCESS;
    printf("Worker Exit\n");
    pthread_exit (&statusWorks[id]);
}

static void printUsage (char *cmdName)
{
    fprintf (stderr, "\nSynopsis: %s OPTIONS [filename]\n"
                     " OPTIONS:\n"
                     " -h --- print this help\n"
                     " -f --- filename\n"
                     , cmdName);
}