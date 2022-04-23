/**
 *  \file worker.h
 *
 *  \brief Assignment 1 : Problem 1 - Number of Words, Number of Words starting with a Vowel and Number of Words ending with a Consonant
 *
 *  Functions used by worker in Assignment 1
 *
 *  \author João Soares (93078) e Pedro Silva (93011)
 */

#ifndef PROG1_ASSIGN1_WORKER_H
#define PROG1_ASSIGN1_WORKER_H

/**
 * \brief Process a chunk of text and store the results in Shared Region
 * Obtains the number of words, number of words starting with a vowel and the number of words starting with a consonant.
 *
 * Operation carried out by the workers.
 * @param chunk Chunk of Text to be processed
 */
void processChunk(struct Chunk_text chunk);

#endif //PROG1_ASSIGN1_WORKER_H
