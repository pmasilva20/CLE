#include <stdio.h>
#include <stdlib.h>
#include <wchar.h>
#include <locale.h>
#include <stdbool.h>
#include <ctype.h>
#include "./prob1_processing.h"



int problem1(char* filename, int* pNWords, int* pNVowelStartWords, int* pNConsonantEndWord){

    int nWords = *pNWords;
    int nVowelStartWords = *pNVowelStartWords;
    int nConsonantEndWord = *pNConsonantEndWord;

    //State Flags
    bool inWord = false;

    //Read files
    int character;
    int previousCharacter = 0;
    
    FILE* pFile;
    pFile = fopen(filename,"r");

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
        //character = preprocessChar(character);
        //printf("After preprocess:%d\n",character);
        //character = tolower(character);

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
    } while ((character = getc(pFile)) != EOF);
    
    fclose(pFile);

    *pNWords = nWords;
    *pNVowelStartWords = nVowelStartWords;
    *pNConsonantEndWord = nConsonantEndWord;

    return 0;
}

