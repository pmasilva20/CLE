#include <stdio.h>
#include <stdlib.h>
#include <wchar.h>
#include <locale.h>
#include <stdbool.h>
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


int problem1(char* filename, int* pNWords, int* pNVowelStartWords, int* pNConsonantEndWord){

    int nWords = *pNWords;
    int nCharacters = 0;
    int nVowelStartWords = *pNVowelStartWords;
    int nConsonantEndWord = *pNConsonantEndWord;

    //State Flags
    bool whiteSpaceFlag = false;
    bool separationFlag = false;
    bool punctuationFlag = false;
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
    while( (character = getc(pFile)) != EOF){
        whiteSpaceFlag = false;
        separationFlag = false;
        punctuationFlag = false;
        
        //Determine how many bytes need to be read in UTF-8
        int bytesNeeded = detectBytesNeeded(character);

        char utf8Char[bytesNeeded+1];
        for (int i = 0; i < bytesNeeded+1; i++)
        {
            utf8Char[i] = 0;
        }
        //Push first byte to most significant byte position and insert another byte read
        //An array is declared so we can print to cmd utf-8 character read
        utf8Char[0] = (char)character;
        for (int i = 0; i < bytesNeeded - 1; i++) {
            int new_char = getc(pFile);
            if(new_char == EOF)break;
            utf8Char[i+1] = (char)new_char;
            character = (character << 8) | new_char;
        }
        if(bytesNeeded == 0){
            printf("Read:%c\n",utf8Char[0]);
        }
        else{
            printf("Read UTF-8:%s\n",utf8Char);
        }

        //printf("Before:%d ",character);
        character = preprocessChar(character);
        //printf("After preprocess:%d\n",character);
        character = checkUpperCase(character);

    


        //Detect if white space
        if(character == ' ' || character == 0x9 || character == 0xA){
            whiteSpaceFlag = true;
        }
        //Detect if separation symbol
        if(character == '-' || character == '"' || character == '['
        || character == ']' || character == '(' || character == ')'){
            separationFlag = true;
        }
        //Detect if punctuation symbol
        if(character == '.' || character == ',' || character == ':' ||
        character == ';' || character == '?' || character == '!'){
            punctuationFlag = true;
        }

        if(whiteSpaceFlag || separationFlag || punctuationFlag){
            if(inWord){
                inWord = false;
                nWords+=1;
                nCharacters = 0;
                if(previousCharacter != 0 && checkConsonants(previousCharacter)){
                    printf("Ending with conconant detected %c\n",previousCharacter);
                    nConsonantEndWord+=1;
                }
                //End of word detected
                printf("End of word\n");
                printf("\n");
            }
        }
        else{
            if(checkVowels(character)){
                if(nCharacters == 0){
                    printf("Starting with vowel detected\n");
                    nVowelStartWords+=1;
                }
            }
            if(checkVowels(character) || checkConsonants(character) || character == '_'){
                inWord = true;
                nCharacters+=1;
            }
        }
        previousCharacter = character;
    }
    previousCharacter = character;
    //Last word, has no way to end before EOF
    if(inWord){
        nCharacters = 0;
        nWords+=1;
        if(checkConsonants(previousCharacter)){
                nConsonantEndWord+=1;
            }
    }
    fclose(pFile);

    *pNWords = nWords;
    *pNVowelStartWords = nVowelStartWords;
    *pNConsonantEndWord = nConsonantEndWord;

    return 0;
}

