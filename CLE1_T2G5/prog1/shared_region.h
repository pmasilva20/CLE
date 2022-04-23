/**
 *  \file fifo.h
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
#ifndef PROG1_SHARED_REGION_H
#define PROG1_SHARED_REGION_H

#include "structures.h"
#include <stdbool.h>

/**
 * \brief Retrieve a stored Text Chunk to be processed.
 *
 * Operation carried out by the workers.
 * @param fileId File Id of the file that is to be retrieved
 * @return File_text structure with file information
 */
struct Chunk_text* getChunkText();

/**
 * \brief Insert a Text Chunk into Shared Region
 * @param chunk Chunk to be inserted
 */
extern int putChunkText(struct Chunk_text chunk);

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
void putFileText(int nWords, int nVowelStartWords, int nConsonantEndWord, int fileID, char* filename);

/**
 * \brief Retrieve a stored File information after all chunks have been processed.
 *
 * Operation carried out by main.
 * @param fileId File Id of the file that is to be retrieved
 * @return File_text structure with file information
 */
struct File_text* getFileText(int fileId);

/**
 * \brief Signal that all text files have been read by main and divided into chunks.
 * Signals any awaiting worker thread.
 *
 * Operation carried out by main.
 */
void finishedProcessingChunks();

/**
 * \brief Check if there are any chunks in Shared Region to process or if main is still processing new chunks.
 * If there are no chunks in Shared Region but main is still processing them, then it waits until a chunks is put.
 *
 * Operation carried out by the workers.
 * @return True if there are chunks to be processed still
 */
bool hasChunksLeft();

#endif //PROG1_SHARED_REGION_H
