/**
 *  \file main.c
 *
 *  \brief Assignment 2 : Problem 1 -- Number of Words, Number of Words starting with a Vowel and Number of Words ending with a Consonant
 *
 *  Description
 *  MPI implementation.
 *
 *
 *  Usage:
 *      \li mpicc -Wall main.c structures.h probConst.h utils.c utils.h sharedRegion.c sharedRegion.h prob1_processing.h prob1_processing.c -o prob1 -lpthread -lm
 *      \li mpiexec -n <number_processes> ./prob1 <file_name1> <file_name2> ...
 *      \li Example: mpiexec -n 4 ./prob1 text0.txt text1.txt text2.txt text3.txt text4.txt 
 *
 *  \author João Soares (93078) & Pedro Silva (93011)
 */

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include "structures.h"
#include <unistd.h>
#include "utils.h"
#include <time.h>
#include <pthread.h>
#include "prob1_processing.h"
#include "probConst.h"
#include "sharedRegion.h"

/* General definitions */

#define WORKTODO 1
#define NOMOREWORK 0


/** \brief Number of processes in the Group */
int rank; 

/** \brief  Group size */
int totProc;

/** \brief Threads return status array */
int *statusDispatcherThreads;

/** \brief Number of files to process */
int totalFiles;

/** \brief Number of chunks made by Thread Read Files */
int totalChunksMade = 0;

/** \brief Copy of filenames provided in argc */
char** fileNames;

/** \brief Array of structs with results for each file processed */
struct FileText* allFilesInfo;

/** \brief Boolean flag for when all files have been processed into chunks and put in Shared Region */
bool madeAllChunks = false;


/** \brief Thread Read Files of the File life cycle routine */
static void *readFileChunks (void *id);

/** \brief Thread Send Chunks and Receive Results life cycle routine */
static void *sendChunksReceiveResults (void *id);

/**
 * \brief Main Function
 *
 * Instantiation of the processing configuration.
 *
 * \param argc number of words of the command line
 * \param argv list of words of the command line
 * \return status of operation
 */

int main(int argc, char *argv[])
{
    /** Initialise MPI and ask for thread support */
    int provided;

    statusDispatcherThreads = malloc(sizeof(int)*2);

    /** Dispatcher Threads internal thread id array */
    pthread_t tIdThreads[2];

    /** Dispatcher Threads defined thread id array */
    unsigned int threadsDispatcher[2];

    /** Pointer to execution status */
    int *status_p;
    
    for (int i = 0; i < 2; i++)
        threadsDispatcher[i] = i;

    srandom ((unsigned int) getpid ());


    MPI_Init_thread (&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &totProc);

    if(provided < MPI_THREAD_MULTIPLE)
    {
        printf("The threading support level is lesser than that demanded.\n");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }


    /* Check if more than 2 processes a Dispatcher and one Worker exist*/
    if (totProc < 2)
    {
        printf("Requires at least two processes - one worker.\n");
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    /**
     * \brief Dispatcher process it is the first process of the group
     */
    if (rank == 0)
    {
        /** time limits **/
        struct timespec start, finish;

        /* command */
        unsigned int whatToDo;

        /* Counting variable */
        unsigned int n;

        /* Counting variable */
        unsigned int r;

        printf("Number of Worker: %d \n", totProc - 1);


        /** Begin of Time measurement */
        clock_gettime(CLOCK_MONOTONIC_RAW, &start);

        /* Check running parameters and load name into memory */
        if (argc < 2)
        {
            printf("No file name!\n");
            whatToDo = NOMOREWORK;
            for (n = 1; n < totProc; n++)
                MPI_Send(&whatToDo, 1, MPI_UNSIGNED, n, 0, MPI_COMM_WORLD);
            MPI_Finalize();
            return EXIT_FAILURE;
        }

        /** Allocate statically enough file results structures for all files to be read **/
        totalFiles = argc - 1;
        struct FileText buffer[totalFiles];
        allFilesInfo = &buffer;
        fileNames = argv;


        /** Intialize each file results strcuture */
        for(int i = 0; i < totalFiles; i++){
            allFilesInfo[i].fileId = i;
            allFilesInfo[i].name = argv[i+1];
            allFilesInfo[i].nConsonantEndWord = 0;
            allFilesInfo[i].nVowelStartWords = 0;
            allFilesInfo[i].nWords = 0;
        }

        /** Generation of Thread to Read and Save Text Chunks of a File **/
        if (pthread_create(&tIdThreads[0], NULL, readFileChunks, &threadsDispatcher[0]) !=0){
            perror("error on creating thread worker");
            exit(EXIT_FAILURE);
        }
        else{
            printf("Thread Read File Chunks Created!\n");
        }

        /** Generation of Thread to Send Matrices to Workers and Receive Partial Results **/
        if (pthread_create(&tIdThreads[1], NULL, sendChunksReceiveResults, &threadsDispatcher[1]) !=0){
            perror("error on creating Thread");
            exit(EXIT_FAILURE);
        }
        else{
            printf("Thread Send Chunks Receive Partial Results Created!\n");
        }

        /** Waiting for the termination of the Dispatcher Threads */
        for (int i = 0; i < 2; i++)
        { if (pthread_join (tIdThreads[i], (void *) &status_p) != 0)
            {
                fprintf(stderr, "Thread %u : error on waiting for thread ",i);
                perror("");
                exit (EXIT_FAILURE);
            }
            else{
                printf ("Thread %u : has terminated with status: %d \n", i, *status_p);
            }
        }

        /** Free any memory used to store chunks **/
        freeChunks();


        /** End of measurement */
        clock_gettime(CLOCK_MONOTONIC_RAW, &finish);

        /** Print Final Results */
        for(int i = 1; i < argc; i++){
            printResults(allFilesInfo[i-1]);
        }

        /** Print Elapsed Time */
        printf("\nElapsed time = %.6f s\n", (finish.tv_sec - start.tv_sec) / 1.0 + (finish.tv_nsec - start.tv_nsec) / 1000000000.0);
    }

    else
    {
        /** \brief  Worker Processes the remainder processes of the group **/

        printf("Worker %d starting up\n", rank);
        /* command */
        unsigned int whatToDo;

        /** Chunk Value */
        struct ChunkText val;

        while (true)
        {

            /** Receive indication of existance of work or not */
            MPI_Recv(&whatToDo, 1, MPI_UNSIGNED, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            /** If no more work, worker terminates */
            if (whatToDo == NOMOREWORK)
            {
                break;
            }

            /** Receive text chunk structure with dinamically allocated chunks array and filename
             * Requires malloc of buffers before hand, requires various Sends and Receives **/

            int fileNameCount;
            int chunkCount;
            int fileId;
            char* filenameReceived;
            int* chunkReceived;


            //Receive size filename
            MPI_Recv(&fileNameCount, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            //Receive fileId
            MPI_Recv(&fileId, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            //Receive size chunk
            MPI_Recv(&chunkCount, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            //Receive chunk
            chunkReceived = malloc(sizeof(int) * chunkCount);
            MPI_Recv(chunkReceived, chunkCount, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            //Receive filename
            filenameReceived = malloc(sizeof(char) * fileNameCount);
            MPI_Recv(filenameReceived, fileNameCount, MPI_CHAR, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);



            val.chunk = chunkReceived;
            val.count = chunkCount;
            val.fileId = fileId;
            val.filename = filenameReceived;
            
            /** Process a chunk and get metrics */
            struct ChunkResults results = processChunk(val);

            /** Send Partial Results for a File to Thread Send Chunks and Receive Results**/
            MPI_Send(&results, sizeof(struct ChunkResults), MPI_BYTE, 0, 0, MPI_COMM_WORLD);
        }
        printf("Worker %d has ended\n", rank);
    }

    MPI_Finalize();

    return EXIT_SUCCESS;
}

/**
 * \brief Function Read Files and make Chunks
 * Each Chunks is stored in Shared Region
 * \param par pointer to application defined worker identification
 */
static void *readFileChunks (void *par){

    /* Pointer to the text stream associated with the file name */
    FILE *f;

    /** Thread ID */
    unsigned int id = *((unsigned int *) par);

    //Read files, make chunks, store them in Shared Region
    for(int i = 0; i < totalFiles; i++){
        char* filename = fileNames[i+1];

        printf("Dispatcher: using file %s",filename);
        

        if ((f = fopen(filename, "r")) == NULL)
        {
            perror("error on file opening for reading");
            MPI_Finalize();
            exit(EXIT_FAILURE);
        }

        // Makes chunks and put in Shared Region
        int totalChunks = makeChunks(f, filename, i);

        totalChunksMade += totalChunks;

        /** Close File */
        fclose(f);
    }
    //Flag that all files have been processed into chunks
    madeAllChunks = true;

    printf("Thread readFileChunks has finished\n");
    statusDispatcherThreads[id] = EXIT_SUCCESS;
    pthread_exit (&statusDispatcherThreads[id]);
}

/**
 * \brief Function Send Chunks and Receive Results
 * Pools for Chunks in Shared Region, sends them to workers to be processed, receives partial results
 * \param par pointer to application defined worker identification
 */
static void *sendChunksReceiveResults (void *par){
     /** Thread ID */
    unsigned int id = *((unsigned int *) par);

    /** Number of Chunks sent*/
    int numberChunksSent=0;

    /** Variable to command if there is work to do or not  */
    unsigned int whatToDo;

    int lastWorker=1;
    
    while(!madeAllChunks || numberChunksSent < totalChunksMade){

        /** Get a Chunk from Shared Region and sent them to workers */
        while(true){
            if (numberChunksSent==totalChunksMade){
                break;
            }

            //Obtain Chunk from SharedRegion
            struct ChunkText chunk;
            getChunks(&chunk,id);
            
            /**There is Work to do **/
            whatToDo=WORKTODO;

            //Send chunk taking into account strcuture used
            sendChunkText(chunk, whatToDo, lastWorker);


            /** Update last Worker that received work*/
            lastWorker = (lastWorker + 1) % totProc;
            if(lastWorker == 0) lastWorker = 1;
            
            /** Update Number of work sent*/
            numberChunksSent++;
        }
    }

    lastWorker = 1;
    printf("Thread sendChunks dispatcher: trying to receive chunk results for %d chunks in total\n",numberChunksSent);
    /** Receive partial results for each file, add them to file results structure */
    for (int i = 0; i < numberChunksSent; i++)
    {
        struct ChunkResults chunkResultsReceived;

        MPI_Recv(&chunkResultsReceived, sizeof(struct ChunkResults), MPI_BYTE, lastWorker, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        lastWorker = (lastWorker + 1) % totProc;
        if(lastWorker == 0) lastWorker = 1;

        struct FileText *file_info = &allFilesInfo[chunkResultsReceived.fileId];
        file_info->nConsonantEndWord += chunkResultsReceived.nConsonantEndWord;
        file_info->nVowelStartWords += chunkResultsReceived.nVowelStartWords;
        file_info->nWords += chunkResultsReceived.nWords;

    }
    printf("Dispatcher %u : Got all results\n", rank);

    /** Inform Workers that there is no more work - Workers will end */
    whatToDo = NOMOREWORK;
    for (int r = 1; r < totProc; r++)
    {
        MPI_Send(&whatToDo, 1, MPI_UNSIGNED, r, 0, MPI_COMM_WORLD);
        printf("Dispatcher ending worker %d : Ending\n", r);
    }

    printf("Thread sendChunksReceiveResults has finished\n");
    statusDispatcherThreads[id] = EXIT_SUCCESS;
    pthread_exit (&statusDispatcherThreads[id]);
}
