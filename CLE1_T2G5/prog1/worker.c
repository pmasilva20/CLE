#include <stdbool.h>
#include <stdio.h>
#include <pthread.h>
#include <stdlib.h>
#include "structures.h"
#include "fifo.h"
#include "prob1_processing.h"


void processChunk(struct Chunk_text chunk){
    //Do prob1 processing
    int nWords = 0;
    int nVowelStartWords = 0;
    int nConsonantEndWord = 0;

    //State Flags
    bool inWord = false;

    //Read files
    int character;
    int previousCharacter = 0;

    for(int i = 0; i < chunk.count; i++){
        character = chunk.chunk[i];

        //Check if inWord
        if(inWord){
            //If white space or separation or punctuation simbol -> inWord is False
            //if lastchar is consonant
            if(checkForSpecialSymbols(character)){
                inWord = false;
                if(checkConsonants(previousCharacter)){
                    nConsonantEndWord+=1;
                }
            }
            // alpha or ` or ' or ‘ or ’
            //If alphanumeric character or underscore or apostrophe -> nothing
            //lastChar = character
            else if(checkVowels(character)
                    || checkConsonants(character)
                    || (character >= '0' && character <= '9')
                    || checkForContinuationSymbols(character)
                    || character == '_'){
                previousCharacter = character;
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
                nWords +=1;
                if(checkVowels(character)){
                    nVowelStartWords+=1;
                }
                previousCharacter = character;
            }
        }
    }
    printf("\n");
    printf("Processed a chunk\n");
    printf("For File %d Nwords %d\n",chunk.fileId,nWords);
    printf("For File %d NVowelwords %d\n",chunk.fileId,nVowelStartWords);
    printf("For File %d NConsonantswords %d\n",chunk.fileId,nConsonantEndWord);
    printf("\n");

    //Put to fifo.c
    putFileText(nWords, nVowelStartWords, nConsonantEndWord, chunk.fileId);
}


void *worker (void *par)
{
    unsigned int id = *((unsigned int *) par);                                                          /* consumer id */

    printf("Worker %d ready!\n",id);

    while(hasChunksLeft()){
        //Get chunk
        struct Chunk_text* chunk = getChunkText();
        if(chunk != NULL)processChunk(*chunk);
        printf("Worker:%d Remaining chunksToProcess %d\n",id,getChunkCount());
    }
    //statusWorks[id] = EXIT_SUCCESS;
    printf("Worker Exit\n");
    //pthread_exit (&statusWorks[id]);
    pthread_exit (EXIT_SUCCESS);
}

