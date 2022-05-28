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
#include "prob1_processing.h"

/* General definitions */

#define WORKTODO 1
#define NOMOREWORK 0

/** \brief Number of characters(encoded in UTF-8) per chunk */
#define chunkSize 20

struct ChunkResults processChunk(struct ChunkText chunk);
int makeChunks(FILE *pFile, char *filename, int fileId, int workerNumber, unsigned int whatToDo);
void sendChunkText(struct ChunkText chunk, unsigned int whatToDo, int n);

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

    /*Number of processes in the Group*/
    int rank;

    /* Group size */
    int totProc;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &totProc);

    /*processing*/
    if (totProc < 2)
    {
        printf("Requires at least two processes - one worker.\n");
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    /**
     * \brief Dispatcher process it is the frist process of the group
     */
    if (rank == 0)
    {

        /** time limits **/
        struct timespec start, finish;

        /* Pointer to the text stream associated with the file name */
        FILE *f;

        /* command */
        unsigned int whatToDo;

        /* Counting variable */
        unsigned int n;

        /* Counting variable */
        unsigned int r;

        printf("Number of Worker: %d \n", totProc - 1);

        /** Initialization FileText*/
        struct FileText file_info;
        file_info.fileId = 0;
        file_info.name = argv[1];
        file_info.nConsonantEndWord = 0;
        file_info.nVowelStartWords = 0;
        file_info.nWords = 0;

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

        if ((f = fopen(argv[1], "r")) == NULL)
        {
            perror("error on file opening for reading");
            whatToDo = NOMOREWORK;
            for (n = 1; n < totProc; n++)
                MPI_Send(&whatToDo, 1, MPI_UNSIGNED, n, 0, MPI_COMM_WORLD);
            MPI_Finalize();
            exit(EXIT_FAILURE);
        }

        // READ M CHUNKS OF DATA TO BE SENT

        // Makes chunks and sends them to totProc workers
        printf("Dispatcher: entering make chunks\n");
        whatToDo = WORKTODO;
        int totalChunksSent = makeChunks(f, argv[1], 0, totProc, whatToDo);

        printf("Dispatcher: All chunks made\n");

        // For each chunk, go though workers and get result
        int lastWorker = 1;
        for (int i = 0; i < totalChunksSent; i++)
        {
            printf("Dispatcher: trying to receive chunk results for %d chunks in total\n",totalChunksSent);
            struct ChunkResults chunkResultsReceived;

            MPI_Recv(&chunkResultsReceived, sizeof(struct ChunkResults), MPI_BYTE, lastWorker, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            lastWorker = (lastWorker + 1) % totProc;
            if(lastWorker == 0) lastWorker = 1;

            /** Fill in struct for file **/
            file_info.nConsonantEndWord += chunkResultsReceived.nConsonantEndWord;
            file_info.nVowelStartWords += chunkResultsReceived.nVowelStartWords;
            file_info.nWords += chunkResultsReceived.nWords;

            printf("Dispatcher %u : File %u Partial result from worker %d\n", rank, chunkResultsReceived.fileId, lastWorker);
        }
        printf("Dispatcher %u : Got all results\n", rank);



        /** CLose File */
        fclose(f);

        /** Inform Workers that there is no more work - Workers will end */
        whatToDo = NOMOREWORK;
        for (r = 1; r < totProc; r++)
        {
            MPI_Send(&whatToDo, 1, MPI_UNSIGNED, r, 0, MPI_COMM_WORLD);
            printf("Dispatcher ending worker %d : Ending\n", r);
        }

        /** End of measurement */
        clock_gettime(CLOCK_MONOTONIC_RAW, &finish);

        /** Print Final Results */
        printResults(file_info);

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
            MPI_Recv(&fileNameCount, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            //Receive fileId
            MPI_Recv(&fileId, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            //Receive size chunk
            MPI_Recv(&chunkCount, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            //Receive chunk
            chunkReceived = malloc(sizeof(int) * 100);
            MPI_Recv(chunkReceived, chunkCount, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            //Receive filename
            filenameReceived = malloc(sizeof(char) * 100);
            MPI_Recv(filenameReceived, fileNameCount, MPI_CHAR, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);



            val.chunk = chunkReceived;
            val.count = chunkCount;
            val.fileId = fileId;
            val.filename = filenameReceived;
            
            printf("Worker %d receive chunk\n", rank);
            /** Results of chunking to be sent */
            struct ChunkResults results = processChunk(val);
            printf("Worker %d processed chunk\n",rank);

            /** Send Partial Result for File **/
            MPI_Send(&results, sizeof(struct ChunkResults), MPI_BYTE, 0, 0, MPI_COMM_WORLD);
            printf("Worker %u processed chunk for File %u\n", rank, results.fileId);
        }
    }

    MPI_Finalize();

    return EXIT_SUCCESS;
}

int makeChunks(FILE *pFile, char *filename, int fileId, int workerNumber, unsigned int whatToDo)
{
    int lastWorker = 1;

    /** Chunk variable being made*/
    struct ChunkText chunk;

    /** State Flags */
    bool inWord = false;

    /** Current character being read */
    int character;

    /** Total number of chunks made **/
    int chunkTotal = 0;

    /** Size of current chunk, maybe be highter then chunkSize **/
    int chunkCount = 0;
    /** Malloc initial minimum size of chunk */
    int *pChunkChars = (int *)calloc(chunkSize, sizeof(int));

    if (pFile == NULL)
    {
        printf("Error reading file\n");
        return 1;
    }

    // Character is of type int due to getc() returning EOF which requires more than 1 byte
    character = getc(pFile);
    if (character == EOF)
    {
        return 1;
    }
    do
    {
        /** Determine how many bytes need to be read in UTF-8 */
        int bytesNeeded = detectBytesNeeded(character);

        /** Push first byte to most significant byte position and insert another byte read */
        for (int i = 0; i < bytesNeeded - 1; i++)
        {
            int new_char = getc(pFile);
            if (new_char == EOF)
                break;
            character = (character << 8) | new_char;
        }

        /** Store character in chunk */
        if (chunkCount < chunkSize)
        {
            pChunkChars[chunkCount] = character;
            chunkCount = chunkCount + 1;
        }
        else
        {
            /** Reallocate 4 more byte of memory, we do this until current word has finished being stored */
            chunkCount = chunkCount + 1;
            int *newPChunkChars = realloc(pChunkChars, chunkCount * sizeof(int));
            pChunkChars = newPChunkChars;
            pChunkChars[chunkCount - 1] = character;
        }

        if (inWord)
        {
            /** Word has ended after encountering a special symbol */
            if (checkForSpecialSymbols(character))
            {
                inWord = false;
                /** After having read chunkSize or more, finish chunk and store it in Shared Region */
                if (chunkCount >= chunkSize)
                {
                    chunk.chunk = pChunkChars;
                    chunk.fileId = fileId;
                    chunk.count = chunkCount;
                    chunk.filename = filename;
                    // Send the chunk with MPI
                    sendChunkText(chunk, whatToDo, lastWorker);
                    lastWorker = (lastWorker + 1) % workerNumber;
                    if(lastWorker == 0) lastWorker = 1;
                    chunkTotal++;

                    // old func putChunkText(chunk);
                    chunkCount = 0;
                    /** Allocated more memory for next chunk */
                    pChunkChars = (int *)calloc(chunkSize, sizeof(int));
                }
            }
        }
        else
        {
            /** Found the start of a new word */
            if (checkVowels(character) || checkConsonants(character) || (character >= '0' && character <= '9') || character == '_')
            {
                inWord = true;
            }
        }
    } while ((character = getc(pFile)) != EOF);

    /** Store last read chunk before finishing reading the Text File */
    chunk.chunk = pChunkChars;
    chunk.fileId = fileId;
    chunk.count = chunkCount;
    chunk.filename = filename;
    // Send the chunk with MPI
    sendChunkText(chunk, whatToDo, lastWorker);
    lastWorker = (lastWorker + 1) % workerNumber;
    if(lastWorker == 0) lastWorker = 1;
    chunkTotal++;
    // old putChunkText(chunk);

    return chunkTotal;
}

void sendChunkText(struct ChunkText chunk, unsigned int whatToDo, int n)
{

    MPI_Send(&whatToDo, 1, MPI_UNSIGNED, n, 0, MPI_COMM_WORLD);

    //Send size filename
    int filenameCount = strlen(chunk.filename);
    MPI_Send(&filenameCount, 1, MPI_INT, n, 0, MPI_COMM_WORLD);
    //Send fileId
    MPI_Send(&chunk.fileId, 1, MPI_INT, n, 0, MPI_COMM_WORLD);
    //Send size chunk
    MPI_Send(&chunk.count, 1, MPI_INT, n, 0, MPI_COMM_WORLD);
    //Send chunk
    MPI_Send(chunk.chunk, chunk.count, MPI_INT, n, 0, MPI_COMM_WORLD);
    //Send filename
    MPI_Send(chunk.filename, filenameCount, MPI_CHAR, n, 0, MPI_COMM_WORLD);

    printf("Dispatcher: Sent chunk to worker %d\n", n);
}

struct ChunkResults processChunk(struct ChunkText chunk)
{
    printf("Starting up process\n");
    /** Results Variables */
    int nWords = 0;
    int nVowelStartWords = 0;
    int nConsonantEndWord = 0;

    /** State Flags */
    bool inWord = false;

    /** Current character being read from chunk */
    int character;
    /** Previous character being read from chunk */
    int previousCharacter = 0;

    for (int i = 0; i < chunk.count; i++)
    {
        character = chunk.chunk[i];

        if (inWord)
        {
            if (checkForSpecialSymbols(character))
            {
                inWord = false;
                if (checkConsonants(previousCharacter))
                {
                    nConsonantEndWord += 1;
                }
            }
            else if (checkVowels(character) || checkConsonants(character) || (character >= '0' && character <= '9') || checkForContinuationSymbols(character) || character == '_')
            {
                previousCharacter = character;
            }
        }
        else
        {
            if (checkVowels(character) || checkConsonants(character) || (character >= '0' && character <= '9') || character == '_')
            {
                inWord = true;
                nWords += 1;
                if (checkVowels(character))
                {
                    nVowelStartWords += 1;
                }
                previousCharacter = character;
            }
        }
    }

    printf("Finishing processing\n");
    struct ChunkResults results;
    results.filename = chunk.filename;
    results.fileId = chunk.fileId;
    results.nWords = nWords;
    results.nVowelStartWords = nVowelStartWords;
    results.nConsonantEndWord = nConsonantEndWord;

    // Send results to main back
    return results;
}
