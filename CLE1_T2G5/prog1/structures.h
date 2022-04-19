#ifndef PROG1_STRUCTURES_H
#define PROG1_STRUCTURES_H


struct File_text{
    char* name;
    int nWords;
    int nVowelStartWords;
    int nConsonantEndWord;
    int id;
};

struct Chunk_text{
    int fileId;
    int* chunk;
    int count;
};


#endif