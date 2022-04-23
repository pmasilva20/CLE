#ifndef PROG1_FIFO_H
#define PROG1_FIFO_H

#include "structures.h"
#include <stdbool.h>


struct Chunk_text* getChunkText();
extern int putChunkText(struct Chunk_text chunk);
int getChunkCount();
void putFileText(int nWords, int nVowelStartWords, int nConsonantEndWord, int fileID, char* filename);
struct File_text* getFileText(int fileId);
void finishedProcessingChunks();
bool hasChunksLeft();

#endif //PROG1_FIFO_H
