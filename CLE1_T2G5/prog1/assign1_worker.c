/**
 *  \file worker.c
 *
 *  \brief Assignment 1 : Problem 1 - Number of Words, Number of Words starting with a Vowel and Number of Words ending with a Consonant
 *
 *  Functions used by worker in Assignment 1
 *
 *  \author João Soares (93078) e Pedro Silva (93011)
 */

#include <stdbool.h>
#include "structures.h"
#include "shared_region.h"
#include "prob1_processing.h"

/**
 * \brief Process a chunk of text and store the results in Shared Region
 * Obtains the number of words, number of words starting with a vowel and the number of words starting with a consonant.
 *
 * Operation carried out by the workers.
 * @param chunk Chunk of Text to be processed
 */
void processChunk(struct Chunk_text chunk){

    //Results Variables
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
//    printf("\n");
//    printf("Processed a chunk\n");
//    printf("For File %d Nwords %d\n",chunk.fileId,nWords);
//    printf("For File %d NVowelwords %d\n",chunk.fileId,nVowelStartWords);
//    printf("For File %d NConsonantswords %d\n",chunk.fileId,nConsonantEndWord);
//    printf("\n");

    //Put to fifo.c
    putFileText(nWords, nVowelStartWords, nConsonantEndWord, chunk.fileId, chunk.filename);
}
