#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <libgen.h>
#include <unistd.h>
#include <string.h>
#include <pthread.h>
#include <ctype.h>
#include <stdbool.h>
#include "./preprocessing.h"
#include "structures.h"
#include "fifo.h"

//Read file and return chunk structure or put in SH direct
//Returns num of chunks made for file
int makeChunks(char* filename,int fileId, int chunkSize){
    struct Chunk_text chunk;

    //State Flags
    bool inWord = false;

    //Read files
    int character;

    FILE* pFile;
    pFile = fopen(filename,"r");
    int chunkCount = 0;
    //Initial malloc for a single chunk
    int* pChunkChars = (int*) malloc(sizeof(int) * chunkSize);

    if(pFile == NULL){
        printf("Error reading file\n");
        return 1;
    }

    //Character is of type int due to EOF having more than 1 byte
    character = getc(pFile);
    if(character == EOF){
        fclose(pFile);
        return 1;
    }
    do{
        //Determine how many bytes need to be read in UTF-8
        int bytesNeeded = detectBytesNeeded(character);

        //Push first byte to most significant byte position and insert another byte read
        //An array is declared so we can print to cmd utf-8 character read
        for (int i = 0; i < bytesNeeded - 1; i++) {
            int new_char = getc(pFile);
            if(new_char == EOF)break;
            character = (character << 8) | new_char;
        }

        //printf("Before:%d ",character);
        character = preprocessChar(character);
        //printf("After preprocess:%d\n",character);
        character = tolower(character);

        //Store character in chunk array
        if(chunkCount < chunkSize){
            printf("%d vs %d\n",chunkCount,chunkSize);
            pChunkChars[chunkCount] = character;
            chunkCount++;
        }
        else{
            //TODO:look at this
            //Realloc 4 more byte of memory, we do this until word finishes
            printf("Reallock\n");
            printf("%d kill %d\n",chunkCount,chunkSize);
            chunkCount = chunkCount + 1;
            int* newPChunkChars = realloc(pChunkChars,   chunkCount * sizeof(int));
            pChunkChars = newPChunkChars;

        }

        //Check of word has finished, if so and chunkSize has been exceeded then we stop this chunk
        if(inWord){
            //If white space or separation or punctuation simbol -> inWord is False
            if(checkForSpecialSymbols(character)){
                inWord = false;
                //Have read either chunksize or more, can finish chunk
                if(chunkCount >= chunkSize){
                    chunk.chunk = pChunkChars;
                    chunk.fileId = fileId;
                    chunk.count = chunkCount;
                    //TODO:Save to either an array or just put in SH directrlly
                    putChunkText(chunk);
                    //Alloc mem for next chunk
                    chunkCount = 0;
                    pChunkChars = (int*) malloc(sizeof(int) * chunkSize);
                }

            }
        }
        else{
            //If alphanumeric character or underscore or apostrophe -> inWord is True
            if(isalnum(character) ||  character == '_' || character == '\''
               || character == 0xE28098 || character == 0xE28099){
                inWord = true;
            }
        }
    } while ((character = getc(pFile)) != EOF);

    //Read last chunk before finishing with file
    chunk.chunk = pChunkChars;
    chunk.fileId = fileId;
    chunk.count = chunkCount;
    //TODO:Save to either an array or just put in SH directrlly
    putChunkText(chunk);
    printf("Last\n");

    fclose(pFile);
    return chunkCount;
}
