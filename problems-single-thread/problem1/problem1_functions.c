#include <stdio.h>
#include <stdlib.h>
#include <wchar.h>
#include <locale.h>
#include <stdbool.h>
#include <ctype.h>
#include "./preprocessing.h"



int detectBytesNeeded(int character){
    if(character < 192){
        return 0;
    }
    else if (character < 224)
    {
        return 2;
    }
    else if (character < 240)
    {
        return 3;
    }
    else
    {
        return 4;
    }
}


bool checkForSpecialSymbols(int character){
        //Detect if white/tab/newline space
        if(character == ' ' || character == 0x9 || character == 0xA || character == 0xD){
            return true;
        }
        //Detect if separation symbol
        if(character == '-' || character == '"' || character == '['
        || character == 0xE2809C || character == 0xe2809D
        || character == ']' || character == '(' || character == ')'){
            return true;
        }
        //Detect if punctuation symbol
        if(character == '.' || character == ',' || character == ':' ||
        character == ';' || character == '?' || character == '!'
        || character == 0xE28093 || character == 0xE280A6 || character == 0xe28094){
            return true;
        }
    return false;
}


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

    //TODO:the apostrophe (' 0x27) and single quotation marks (‘ 0xE28098 - ’ 0xE28099) are considered here to merge two words into a single one.



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
            //If alphanumeric character or underscore or apostrophe -> nothing
                //lastChar = character
            if(isalnum(character) ||  character == '_' || character == '\''
                || character == 0xE28098 || character == 0xE28099){
                previousCharacter = character;
            }
        }
        else{
            //If white space or separation or punctuation simbol -> nothing
            //If alphanumeric character or underscore or apostrophe -> inWord is True
                //nWords += 1, checkVowel() -> nWordsBV+=1, lastChar = character
            if(isalnum(character) ||  character == '_' || character == '\''
                || character == 0xE28098 || character == 0xE28099){
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

