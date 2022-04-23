/**
 *  \file assign1_functions.c
 *
 *  \brief Assignment 1 : Problem 1 - Number of Words, Number of Words starting with a Vowel and Number of Words ending with a Consonant
 *
 *  Functions used in Assignment 1 by Main Thread
 *
 *  \author João Soares (93078) e Pedro Silva (93011)
 */

#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include "prob1_processing.h"
#include "structures.h"
#include "shared_region.h"

/**
 * \brief Reads UTF-8 encoded characters from a Text File and makes Chunks out of them
 * Chunks are stored in order so they surpass chunkSize but also end in a non word character
 * All chunks are stored in the Shared Region
 * @param filename Filename of the Text File to be read
 * @param fileId File id that identifies this file
 * @param chunkSize Minimum num of characters in each chunk
 * @return Num of chunks made
 */
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
    int* pChunkChars = (int*) calloc(chunkSize, sizeof(int));

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
        for (int i = 0; i < bytesNeeded - 1; i++) {
            int new_char = getc(pFile);
            if(new_char == EOF)break;
            character = (character << 8) | new_char;
        }

        //Store character in chunk array
        if(chunkCount < chunkSize){
            //printf("Puttin in chunk %d vs %d\n",chunkCount,chunkSize);
            pChunkChars[chunkCount] = character;
            chunkCount = chunkCount + 1;
        }
        else{
            //Reallocate 4 more byte of memory, do this until current word has finished being stored
            //printf("Realloc Have %d need %d\n",chunkCount,chunkSize);
            chunkCount = chunkCount + 1;
            int* newPChunkChars = realloc(pChunkChars,   chunkCount * sizeof(int));
            pChunkChars = newPChunkChars;
            pChunkChars[chunkCount-1] = character;

        }

        if(inWord){
            if(checkForSpecialSymbols(character)){
                inWord = false;
                //After having read chunkSize or more, finish chunk and store it in Shared Region
                if(chunkCount >= chunkSize){
                    chunk.chunk = pChunkChars;
                    chunk.fileId = fileId;
                    chunk.count = chunkCount;
                    chunk.filename = filename;
                    //printf("Put chunk in\n");
                    putChunkText(chunk);
                    chunkCount = 0;
                    //Allocated more memory for next chunk
                    pChunkChars = (int*) calloc(chunkSize, sizeof(int));
                }
            }
        }
        else{
            if(checkVowels(character)
               || checkConsonants(character)
               || (character >= '0' && character <= '9')
               || character == '_'){
                inWord = true;
            }
        }
    } while ((character = getc(pFile)) != EOF);

    //Store last read chunk before finishing reading the Text File
    chunk.chunk = pChunkChars;
    chunk.fileId = fileId;
    chunk.count = chunkCount;
    chunk.filename = filename;
    chunkCount++;
    putChunkText(chunk);

    fclose(pFile);
    return chunkCount;
}
