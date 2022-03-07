
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
int problem1(char* filename, int* pNWords, int* pNCharacters, int* pNConsonants,
            int* pCharFreq, int** pConsonantFreq){

    int nWords = *pNWords;
    int nCharacters = *pNCharacters;
    int nConsonants = *pNConsonants;

    //State Flags
    bool white_space_flag = false;
    bool separation_flag = false;
    bool punctuation_flag = false;
    bool inWord = false;

    //Read files
    int character;
    
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

            //RESIZE pCharFreq
            while(nCharacters > CHAR_FREQ_NUM){
                printf("Resize CharFreq\n");
                pCharFreq = realloc(pCharFreq, CHAR_FREQ_NUM * 2 * sizeof(int));
                //Initialize to 0
                for(int i = CHAR_FREQ_NUM; i < CHAR_FREQ_NUM*2; i++){
                    pCharFreq[i] = 0;
                }
                CHAR_FREQ_NUM *=2;
            }
            pCharFreq[nCharacters-1]+=1;
            pConsonantFreq[nCharacters-1][nConsonants]+=1;
            nCharacters = 0;
            nConsonants = 0;
            nWords+=1;
        }
        else{
            inWord = true;
            nCharacters+=1;
            if(checkConsonants(character)){
                nConsonants+=1;
            }
        }
        printf("\n");
    }
    //Last word, has no way to end before EOF
    if(inWord){

        //RESIZE pCharFreq
        while(nCharacters > CHAR_FREQ_NUM){
           
            int old_char_freq = CHAR_FREQ_NUM;
            printf("Resize CharFreq\n");
            CHAR_FREQ_NUM = resizeCharFreq(pCharFreq,old_char_freq,2);
            printf("Resize ConsonantFreq\n");
            //resizeConsonantFreq(pConsonantFreq,old_char_freq,2);
        }
        pCharFreq[nCharacters-1]+=1;
        pConsonantFreq[nCharacters-1][nConsonants]+=1;

        nCharacters = 0;
        nWords+=1;
    }
    fclose(pFile);

    *pNWords = nWords;
    *pNCharacters = nCharacters;
    *pNConsonants = nConsonants;

    return 0;
}

int main (int argc, char** argv){
    for(int textIdx = 1; textIdx < argc; textIdx++){
        //Vars needed
        int nWords = 0;
        int nCharacters = 0;
        int nConsonants = 0;
        //Array of number of words with n characters, n is index+1 of arr
        //Start with 10, might resize
        int* pCharFreq = (int*) calloc(CHAR_FREQ_NUM,sizeof(int));

        //Double Array of num of words of i1 length and i2 consonants, 
        //first is length of word, second is num of consoants
        int** pConsonantFreq = (int**) calloc(CHAR_FREQ_NUM, sizeof(int*));
        for( int i = 0; i < CHAR_FREQ_NUM; i++){
            pConsonantFreq[i] = (int*) calloc(10,sizeof(int));
        }
        int error_code = problem1(argv[textIdx],&nWords,&nCharacters,&nConsonants,pCharFreq,pConsonantFreq);

        if(error_code != 0){
            printf("Error during file processing of %s",argv[textIdx]);
            continue;
        }

        printf("File %s\n",argv[textIdx]);
        printf("Number of words:%d\n",nWords);
        printf("Number of consonants:%d\n",nConsonants);
        printf("Word Length:\n");
        //Lines
        printf("%-3s", "");
        for( int i = 0; i < CHAR_FREQ_NUM; i++){
            printf("%-5d",i+1);
        }
        printf("\n");
        //CharFreq
        printf("%-3s", "");
        for( int i = 0; i < CHAR_FREQ_NUM; i++){
            printf("%-5d",pCharFreq[i]);
        }
        printf("\n");
        //CharFreq(%)
        printf("%-3s", "");
        for( int i = 0; i < CHAR_FREQ_NUM; i++){
            printf("%-5.1f",((double)pCharFreq[i] / (double)nWords) * 100);
        }
        printf("\n");

        //ConsonantFreq matrix
        for( int i1 = 0; i1 < CHAR_FREQ_NUM; i1++){
            printf("%-3d", i1);
            for(int i2 = 0; i2 < 10; i2++){
                printf("%-5d",pConsonantFreq[i1][i2]);
            }
            printf("\n");
        }

        //Free up allocated memory
        free(pCharFreq);
        for(int i = 0; i < CHAR_FREQ_NUM; i++){
            free(pConsonantFreq[i]);
        }
        free(pConsonantFreq);
    }
}