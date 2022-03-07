
#include "./preprocessing.h" 
#include <stdio.h>
#include <stdlib.h>
#include <wchar.h>
#include <locale.h>
#include <stdbool.h>


//Constants
int CHAR_FREQ_NUM = 10;


//Resize array
int resizeCharFreq(int* array, int arr_size,int factor){
    printf("Before realloc %p\n",(void*)array);
    printf("%d\n",arr_size);
    array = realloc(array, arr_size * factor * sizeof(int));
    printf("After realloc %p\n",(void*)array);
    //Initialize to 0
    for(int i = arr_size; i < arr_size*factor; i++){
        array[i] = 0;
    }
    return arr_size*factor;
}

//Resize array
int resizeConsonantFreq(int** array, int arr_size,int factor){
    printf("Before realloc %p\n",(void*)array);
    array = realloc(array, arr_size * factor * sizeof(int*));
    printf("After realloc %p\n",(void*)array);
    //Initialize to 0
    for(int i = arr_size; i < arr_size*factor; i++){
        //array[i] = 0;
        //Should alloc memory here
    }
    return arr_size*factor;
}

//TODO:Consonants freqs seem wrong
//TODO:CHAR_FREQ_NUM should be a state variable
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

        int* pCharacter = &character;
        setlocale(LC_ALL, "");
        //Find some way of reading utf-8
        printf("Actually Read:%s\n",(char*)pCharacter);
        printf("Read:%d ones\n",ones_counter);
        if(ones_counter == 0){
            //Basic ASCII letter
            printf("ACII letter read");
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
        if(character == 0x20 || character == 0x9 || character == 0xA){
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
                nConsonantEndWord+=1;
            }
        }
        else{
            inWord = true;
            if(!checkConsonants(character)){
                if(nCharacters == 0){
                    printf("%c",(char)character);
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

int main (int argc, char** argv){
    for(int textIdx = 1; textIdx < argc; textIdx++){
        //Vars needed
        int nWords = 0;
        int nVowelStartWords = 0;
        int nConsonantEndWord = 0;

        int error_code = problem1(argv[textIdx],&nWords,&nVowelStartWords,&nConsonantEndWord);

        if(error_code != 0){
            printf("Error during file processing of %s",argv[textIdx]);
            continue;
        }

        printf("File %s\n",argv[textIdx]);
        printf("Number of words:%d\n",nWords);
        printf("Number of words which start with a vowel:%d\n",nVowelStartWords);
        printf("Number of words which end with a consonant:%d\n",nConsonantEndWord);
    }
}