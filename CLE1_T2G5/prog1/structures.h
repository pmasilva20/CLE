/**
 *  \file structures.h
 *
 *  \brief Assignment 1 : Problem 1 - Number of Words, Number of Words starting with a Vowel and Number of Words ending with a Consonant
 *
 *  Main Program
 *
 *  \author João Soares (93078) e Pedro Silva (93011)
 */

#ifndef PROG1_STRUCTURES_H
#define PROG1_STRUCTURES_H

/** Structure of File results */
struct FileText{
    char* name;
    int nWords;
    int nVowelStartWords;
    int nConsonantEndWord;
    int fileId;
};

/** Structure of Chunk of Text from a file */
struct ChunkText{
    char* filename;
    int fileId;
    int* chunk;
    int count;
};


#endif