/**
 *  \file fifo.h
 *
 *  \brief Assignment 2 : Problem 1 - Number of Words, Number of Words starting with a Vowel and Number of Words ending with a Consonant
 *
 *  Shared Region
 *
 *  Dispatcher Operations:
 *      \li putChunkText
 *      \li freeChunks
 *
 *  Workers Operations:
 *      \li getChunks
 *
 *  \author João Soares (93078) e Pedro Silva (93011)
 */
#ifndef PROG1_SHARED_REGION_H
#define PROG1_SHARED_REGION_H

#include "structures.h"
#include <stdbool.h>


/**
 * \brief Insert a Text Chunk into Shared Region
 * @param chunk Chunk to be inserted
 */
extern int putChunkText(struct ChunkText chunk);


/**
 * \brief Check if there are any chunks in Shared Region to process or if main is still processing new chunks.
 * If there are no chunks in Shared Region but main is still processing them, then it waits until a chunks is put.
 * Else retrieves a stored Text Chunk to be processed.
 *
 * Operation carried out by the dispatcher.
 * @return True if there are chunks to be processed still
 */
bool getChunks(struct ChunkText *chunk, unsigned int consId);

/**
 * \brief Free any memory allocated previously in Shared Region.
 *
 * Operation carried out by dispatcher.
 */
void freeChunks();

#endif //PROG1_SHARED_REGION_H
