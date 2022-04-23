/**
 *  \file assign1_functions.h
 *
 *  \brief Assignment 1 : Problem 1 - Number of Words, Number of Words starting with a Vowel and Number of Words ending with a Consonant
 *
 *  Functions used in Assignment by Main Thread
 *
 *  \author João Soares (93078) e Pedro Silva (93011)
 */

#ifndef PROG1_ASSIGN1_FUNCTIONS_H
#define PROG1_ASSIGN1_FUNCTIONS_H

/**
 * \brief Reads UTF-8 encoded characters from a Text File and makes Chunks out of them
 * Chunks are stored in order so they surpass chunkSize but also end in a non word character
 * All chunks are stored in the Shared Region
 * @param filename Filename of the Text File to be read
 * @param fileId File id that identifies this file
 * @param chunkSize Minimum num of characters in each chunk
 * @return Num of chunks made
 */
int makeChunks(char* filename,int fileId, int chunkSize);

#endif //PROG1_ASSIGN1_FUNCTIONS_H
