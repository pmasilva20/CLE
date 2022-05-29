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


int makeChunks(FILE *pFile, char *filename, int fileId, int workerNumber);

void sendChunkText(struct ChunkText chunk, unsigned int whatToDo, int n);

struct ChunkResults processChunk(struct ChunkText chunk);