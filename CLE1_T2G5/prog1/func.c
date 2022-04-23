#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include "prob1_processing.h"
#include "structures.h"
#include "fifo.h"

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
            //TODO:look at this
            //Realloc 4 more byte of memory, we do this until word finishes
            //printf("Realloc Have %d need %d\n",chunkCount,chunkSize);
            chunkCount = chunkCount + 1;
            int* newPChunkChars = realloc(pChunkChars,   chunkCount * sizeof(int));
            pChunkChars = newPChunkChars;
            pChunkChars[chunkCount-1] = character;

        }

        //Check if inWord
        if(inWord){
            //If white space or separation or punctuation simbol -> inWord is False
            //if lastchar is consonant
            if(checkForSpecialSymbols(character)){
                inWord = false;
                //Have read either chunksize or more, can finish chunk
                if(chunkCount >= chunkSize){
                    chunk.chunk = pChunkChars;
                    chunk.fileId = fileId;
                    chunk.count = chunkCount;
                    chunk.filename = filename;
                    //TODO:Save to either an array or just put in SH directrlly
                    //printf("Put chunk in\n");
                    putChunkText(chunk);
                    //Alloc mem for next chunk
                    chunkCount = 0;
                    pChunkChars = (int*) calloc(chunkSize, sizeof(int));
                }
            }
        }
        else{
            //If white space or separation or punctuation simbol -> nothing
            // alpha or ` or ' or ‘ or ’
            //If alphanumeric character or underscore or apostrophe -> inWord is True
            //nWords += 1, checkVowel() -> nWordsBV+=1, lastChar = character
            if(checkVowels(character)
               || checkConsonants(character)
               || (character >= '0' && character <= '9')
               || character == '_'){
                inWord = true;
            }
        }
    } while ((character = getc(pFile)) != EOF);

    //Read last chunk before finishing with file
    chunk.chunk = pChunkChars;
    chunk.fileId = fileId;
    chunk.count = chunkCount;
    chunk.filename = filename;
    chunkCount++;
    //TODO:Save to either an array or just put in SH directrlly
    //printf("Put Last chunk in\n");
    putChunkText(chunk);
    printf("Final Num of chunks %d\n",chunkCount);

    fclose(pFile);
    return chunkCount;
}
