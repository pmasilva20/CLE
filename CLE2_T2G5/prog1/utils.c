/**
 *  \file utils.c 
 *
 *  \brief Assignment 2 : Problem 1 - Counting WOrds
 *
 *  Methods/Operations used by Dispatcher/Workers
 *  
 *  Dispatcher Methods:
 *      \li printResults
 * 
 *
 *  \author João Soares (93078) & Pedro Silva (93011)
*/
#include <stdbool.h>
#include <stdlib.h>
#include <stdio.h>
#include "structures.h"
#include "probConst.h"
#include <mpi.h>
#include "prob1_processing.h"
#include <string.h>
#include "sharedRegion.h"


/**
 * Send a text chunk
 * Needs to be sent separatelly due to use of dinamically allocated memory
 * \param chunk Structure of text chunk to be sent
 * \param whatToDo Command to be sent
 * \param n Worker Id to send to
 */
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

}

/**
 * Process a file into various text chunks, store them in Shared Region
 * \param pFile Pointer to file to be processed
 * \param filename Pointer to char array with filename
 * \param fileId id of file processed
 * \return Number of chunks made
 */
int makeChunks(FILE *pFile, char *filename, int fileId)
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

                    //Store the chunk in FIFO
                    putChunkText(chunk);
                    chunkTotal++;


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

    //Store the chunk in FIFO
    putChunkText(chunk);
    chunkTotal++;

    return chunkTotal;
}


/**
 * Process a text chunk
 * \param chunk Structure of a text chunk
 * \return Structure of Partial File Results with metrics obtained for the chunk processed
 */
struct ChunkResults processChunk(struct ChunkText chunk)
{
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

    struct ChunkResults results;
    results.filename = chunk.filename;
    results.fileId = chunk.fileId;
    results.nWords = nWords;
    results.nVowelStartWords = nVowelStartWords;
    results.nConsonantEndWord = nConsonantEndWord;

    // Send results to main back
    return results;
}

/**
 * Print in the terminal the results stored 
 * \param filesToProcess File Results Structure
 */
void printResults(struct FileText results){

    /** Get all Files statistics and print to console */
    printf("\n");
    printf("File name: %s\n",results.name);
    printf("Total number of words = %d\n",results.nWords);
    printf("N. of words beginning with a vowel = %d\n",results.nVowelStartWords);
    printf("N. of words ending with a consonant = %d\n",results.nConsonantEndWord);

    printf("\n");

}