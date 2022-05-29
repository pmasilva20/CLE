/**
 *  \file utils.h 
 *
 *  \brief Assignment 2 : Problem 1 - Word Count
 *
 *  Methods/Operations used by Dispatcher/Workers
 *  
 *  Dispatcher Methods:
 *      \li printResults
 * 
 *  Worker Methods:
 *      \li makeChunks
 *
 *  \author João Soares (93078) & Pedro Silva (93011)
*/


#include "structures.h"
#include <stdbool.h>
#include <stdio.h>

/**
 * Print in the terminal the results stored 
 * \param filesToProcess Number of Files
 */
extern void printResults(struct FileText results);

/**
 * Process a file into various text chunks, store them in Shared Region
 * \param pFile Pointer to file to be processed
 * \param filename Pointer to char array with filename
 * \param fileId id of file processed
 * \return Number of chunks made
 */
int makeChunks(FILE *pFile, char *filename, int fileId);

/**
 * Send a text chunk
 * Needs to be sent separatelly due to use of dinamically allocated memory
 * \param chunk Structure of text chunk to be sent
 * \param whatToDo Command to be sent
 * \param n Worker Id to send to
 */
void sendChunkText(struct ChunkText chunk, unsigned int whatToDo, int n);

/**
 * Process a text chunk
 * \param chunk Structure of a text chunk
 * \return Structure of Partial File Results with metrics obtained for the chunk processed
 */
struct ChunkResults processChunk(struct ChunkText chunk);