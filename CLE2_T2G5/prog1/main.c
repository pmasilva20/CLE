/**
 *  \file main.c
 *
 *  \brief Assignment 2 : Problem 1 -
 *
 *  Description
 *  MPI implementation.
 *
 *
 *  Usage:
 *      \li mpicc -Wall -o main main.c structures.h  utils.c utils.h
 *      \li mpiexec -n <number_processes> ./main <file_name>
 *      \li Example: mpiexec -n 5 ./main mat128_32.bin
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

int totalFiles;

int totalChunksMade = 0;

char** fileNames;

struct FileText* all_files_info;

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


    /* Check if more than 2 processes Dispatcher and one Worker exist*/
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

        /** All file results Structures **/
        totalFiles = argc - 1;
        struct FileText buffer[totalFiles];
        all_files_info = &buffer;
        fileNames = argv;


        /** Initialization FileText*/
        for(int i = 0; i < totalFiles; i++){
            all_files_info[i].fileId = i;
            all_files_info[i].name = argv[i+1];
            all_files_info[i].nConsonantEndWord = 0;
            all_files_info[i].nVowelStartWords = 0;
            all_files_info[i].nWords = 0;
        }

        //Thread creation
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


        /** End of measurement */
        clock_gettime(CLOCK_MONOTONIC_RAW, &finish);

        /** Print Final Results */
        for(int i = 1; i < argc; i++){
            printResults(all_files_info[i-1]);
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

        int count = 0;



        while (true)
        {
            printf("Worker %d alive %d\n",rank,count);

            /** Receive indication of existance of work or not */
            MPI_Recv(&whatToDo, 1, MPI_UNSIGNED, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            printf("Worker %d receive what to do\n", rank);
            /** If no more work, worker terminates */
            if (whatToDo == NOMOREWORK)
            {
                break;
            }

            
            int fileNameCount;
            int chunkCount;
            int fileId;
            char* filenameReceived;
            int* chunkReceived;


            //Receive size filename
            printf("Worker %d in receive lane\n", rank);
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
            
            //printf("Worker %d receive chunk\n", rank);
            /** Results of chunking to be sent */
            struct ChunkResults results = processChunk(val);
            //printf("Worker %d processed chunk\n",rank);

            /** Send Partial Result for File **/
            MPI_Send(&results, sizeof(struct ChunkResults), MPI_BYTE, 0, 0, MPI_COMM_WORLD);
            printf("Worker %u processed chunk for File %u\n", rank, results.fileId);
            count++;
        }
        printf("Worker %d end\n", rank);
    }

    MPI_Finalize();

    return EXIT_SUCCESS;
}


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

        // Makes chunks and put in SR
        printf("Dispatcher: entering make chunks\n");
        int totalChunks = makeChunks(f, filename, i, totProc);

        totalChunksMade += totalChunks;

        /** Close File */
        fclose(f);
    }
    madeAllChunks = true;
    printf("Dispatcher: All chunks made and put in SR\n");

    printf("Thread readFileChunks has finished\n");
    statusDispatcherThreads[id] = EXIT_SUCCESS;
    pthread_exit (&statusDispatcherThreads[id]);
}

//Pool for Chunks and send them to workers
static void *sendChunksReceiveResults (void *par){
     /** Thread ID */
    unsigned int id = *((unsigned int *) par);

    /** Number of Chunks sent*/
    int numberChunksSent=0;

    /** Variable to command if there is work to do or not  */
    unsigned int whatToDo;

    /** Variables used to verify if the non-blocking operation are complete*/     
    bool allMsgRec, recVal, msgRec[totProc];

    int count1 = 0;
    int count2 = 0;
    int count3 = 0;
    int lastWorker=1;
    
    /** MPI_Request handles for the non-blocking operations of sending Matrices and Receive partial Results */
    //MPI_Request reqSnd[totProc], reqRec[totProc];

    printf("Thread sendChunks getting chunks\n");

    while(!madeAllChunks || numberChunksSent < totalChunksMade){
        /** MPI_Request handler to WhatoDo message */
        MPI_Request reqWhatoDo;

        /** Last Worker to receive work
         * It will save the last worker that receives work.
         * Useful for knowing which workers the dispatcher should expect to receive messages from.
         * **/


        while(true){
            if (numberChunksSent==totalChunksMade){
                break;
            }

            //Obtain Chunk from SharedRegion
            struct ChunkText chunk;
            getChunks(&chunk,id);
            
            /**There is Work to do **/
            whatToDo=WORKTODO;

            if(lastWorker == 1) count1++;
            if(lastWorker == 2) count2++;
            if(lastWorker == 3) count3++;
            if(lastWorker != 1 && lastWorker != 2 && lastWorker != 3){
                printf("LUCIFER TAKE ME\n");
            }

            //Send chunk by subdividing structure
            printf("Counts 1:%d 2:%d 3:%d\n",count1,count2,count3);
            printf("Chunk Sent by T2 - Chunk %d to Worker %d \n",numberChunksSent,lastWorker);
            sendChunkText(chunk, whatToDo, lastWorker);



            
            /** Update last Worker that received work*/
            lastWorker = (lastWorker + 1) % totProc;
            if(lastWorker == 0) lastWorker = 1;
            
            /** Update Number of work sent*/
            numberChunksSent++;
        }
    }

    printf("Thread sendChunks getting results\n");
    //Receive partial Results
    // For each chunk, go though workers and get result

    lastWorker = 1;
    printf("Thread sendChunks dispatcher: trying to receive chunk results for %d chunks in total\n",numberChunksSent);
    for (int i = 0; i < numberChunksSent; i++)
    {
        struct ChunkResults chunkResultsReceived;

        printf("Dispatcher stillAlive %d\n",i);
        MPI_Recv(&chunkResultsReceived, sizeof(struct ChunkResults), MPI_BYTE, lastWorker, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        lastWorker = (lastWorker + 1) % totProc;
        if(lastWorker == 0) lastWorker = 1;

        /** Fill in struct for file **/
        struct FileText *file_info = &all_files_info[chunkResultsReceived.fileId];
        file_info->nConsonantEndWord += chunkResultsReceived.nConsonantEndWord;
        file_info->nVowelStartWords += chunkResultsReceived.nVowelStartWords;
        file_info->nWords += chunkResultsReceived.nWords;

        //printf("Dispatcher %u : File %u Partial result from worker %d\n", rank, chunkResultsReceived.fileId, lastWorker);
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
