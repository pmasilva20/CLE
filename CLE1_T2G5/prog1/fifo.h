//
// Created by pmasilva20 on 19-04-2022.
//

#ifndef PROG1_FIFO_H
#define PROG1_FIFO_H

#include "structures.h"


struct Chunk_text getChunkText();
extern int putChunkText(struct Chunk_text chunk);
int getChunkCount();
void putFileText(int nWords, int nVowelStartWords, int nConsonantEndWord, int fileID);
struct File_text getFileText(int fileId);

#endif //PROG1_FIFO_H
