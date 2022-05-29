/**
 *  \file structures.h
 *
 *  \brief Assignment 1 : Problem 1 - Counting Words
 *
 *  Definition of Structures
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

/** Structure of Chunk Results(Partial File Results)*/
struct ChunkResults{
    char* filename;
    int fileId;
    int nWords;
    int nVowelStartWords;
    int nConsonantEndWord;
};


/** Structure of Chunk of Text from a file */
struct ChunkText{
    char* filename;
    int fileId;
    int* chunk;
    int count;
};


#endif //PROG1_STRUCTURES_H