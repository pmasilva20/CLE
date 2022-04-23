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
 * @param consId Id of worker
 * @param chunk Chunk of Text to be processed
 */
void processChunk(struct ChunkText chunk, unsigned int consId) {

    /** Results Variables */
    int nWords = 0;
    int nVowelStartWords = 0;
    int nConsonantEndWord = 0;

    /** State Flags */
    bool inWord = false;

    /** Current character being read from chunk */
    int character;
    /** Previous character being read from chunk */
    int previousCharacter = 0;

    for(int i = 0; i < chunk.count; i++){
        character = chunk.chunk[i];

        if(inWord){
            if(checkForSpecialSymbols(character)){
                inWord = false;
                if(checkConsonants(previousCharacter)){
                    nConsonantEndWord+=1;
                }
            }
            else if(checkVowels(character)
                    || checkConsonants(character)
                    || (character >= '0' && character <= '9')
                    || checkForContinuationSymbols(character)
                    || character == '_'){
                previousCharacter = character;
            }
        }
        else{
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

    /** Store results in Shared Region */
    putFileText(nWords, nVowelStartWords, nConsonantEndWord, chunk.fileId, chunk.filename, consId);
}
