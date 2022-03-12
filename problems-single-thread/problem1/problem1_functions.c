#include <stdio.h>
#include <stdlib.h>
#include <wchar.h>
#include <locale.h>
#include <stdbool.h>
#include "./preprocessing.h"

int checkUpperCase(int character);

//TODO:Consonants freqs seem wrong
int problem1(char* filename, int* pNWords, int* pNVowelStartWords, int* pNConsonantEndWord){

    int nWords = *pNWords;
    int nCharacters = 0;
    int nVowelStartWords = *pNVowelStartWords;
    int nConsonantEndWord = *pNConsonantEndWord;

    //State Flags
    bool white_space_flag = false;
    bool separation_flag = false;
    bool punctuation_flag = false;
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
    //It's int due to EOF having more than 1 byte
    while( (character = getc(pFile)) != EOF){
        white_space_flag = false;
        separation_flag = false;
        punctuation_flag = false;

        printf("Read from first byte:%c\n",character);
        int ones_counter = 0;

        if(character < 192){
            ones_counter = 0;
        }
        else if (character < 224)
        {
            ones_counter = 2;
        }
        else if (character < 240)
        {
            ones_counter = 3;
        }
        else
        {
            ones_counter = 4;
        }
        //Push first byte to most significant byte position and insert another byte read
        for (int i = 0; i < ones_counter - 1; i++) {
            int new_char = getc(pFile);
            if(new_char == EOF)break;
            character = (character << 8) | new_char;
        }

        //Find some way of reading utf-8
        //printf("Actually Read:%s\n",(char*)pCharacter);
        //printf("Read:%d ones\n",ones_counter);
        if(ones_counter == 0){
            //Basic ASCII letter
            printf("ACII letter read");
            character=checkUpperCase(character);
        }
        else{
            printf("UTF-8 encoding letter\n");
            printf("Before:%d ",character);
            character = preprocessChar(character);
            //PREPROCESSING DONE
            printf("After preprocess:%d\n",character);
        }
        printf("\n");
        //Detect white space
        if(character == ' ' || character == 0x9 || character == 0xA){
            white_space_flag = true;
        }
        //Detect separation symbol
        if(character == '-' || character == '"' || character == '['
        || character == ']' || character == '(' || character == ')'){
            separation_flag = true;
        }
        //Detect punctuation symbol
        if(character == '.' || character == ',' || character == ':' ||
        character == ';' || character == '?' || character == '!'){
            punctuation_flag = true;
        }

        if(white_space_flag || separation_flag || punctuation_flag){
            inWord = false;
            nWords+=1;
            nCharacters = 0;
            if(previousCharacter != 0 && checkConsonants(previousCharacter)){
                printf("Consonant: %c",(char)character);
                nConsonantEndWord+=1;
            }
        }
        else{
            inWord = true;
            if(!checkConsonants(character)){
                if(nCharacters == 0){
                    printf("Vogal: %c",(char)character);
                    nVowelStartWords+=1;
                }
            }
            nCharacters+=1;
        }
        previousCharacter = character;
        printf("\n");
    }
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

