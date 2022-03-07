
#include "./preprocessing.h" 
#include <stdio.h>
#include <stdlib.h>
#include <wchar.h>
#include <locale.h>
#include <stdbool.h>


int main (int argc, char** argv){
    //Constants
    unsigned int CHAR_FREQ_NUM = 10;

    for(int textIdx = 1; textIdx < argc; textIdx++){
        //State Flags
        bool white_space_flag = false;
        bool separation_flag = false;
        bool punctuation_flag = false;
        bool inWord = false;

        //Vars needed
        unsigned int nWords = 0;
        unsigned int nCharacters = 0;
        unsigned int nConsonants = 0;
        //Array of number of words with n characters, n is index+1 of arr
        //Start with 10, might resize
        int* pCharFreq = (int*) calloc(CHAR_FREQ_NUM,sizeof(int));

        //Double Array of num of words of i1 length and i2 consonants, 
        //first is length of word, second is num of consoants
        int** pConsonantFreq = (int**) calloc(CHAR_FREQ_NUM, sizeof(int*));
        for(unsigned int i = 0; i < CHAR_FREQ_NUM; i++){
            pConsonantFreq[i] = (int*) calloc(10,sizeof(int));
        }

        //Read files
        int character;
        
        FILE* pFile;
        pFile = fopen(argv[textIdx],"r");

        if(pFile == NULL){
            printf("Error reading file");
            exit(-1);
        }
        //It's int due to EOF having more than 1 byte
        while( (character = getc(pFile)) != EOF){
            white_space_flag = false;
            separation_flag = false;
            punctuation_flag = false;


            //printf("Read:%d\n",character);
            printf("Read:%c\n",character);
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
            for (int i = 0; i < ones_counter - 1; i++) {
                int new_char = getc(pFile);
                if(new_char == EOF)break;
                //<<8 e depois |
                //printf("Before:%d\n",character);
                character = (character << 8) | new_char;
                //printf("After:%d\n",character);
            }

            int* pCharacter = &character;
            setlocale(LC_ALL, "");
            //This typecasting does not work at all
            printf("Actually Read:%s\n",(char*)pCharacter);
            printf("Read:%d ones\n",ones_counter);
            if(ones_counter == 0){
                //Basic ASCII letter
                printf("ACII letter");
            }
            else{
                printf("UTF-8 encoding letter\n");
                printf("Before:%c ",character);
                character = preprocessChar(character);
                //PREPROCESSING DONE
                printf("After preprocess:%c\n",character);
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
                if(nCharacters > CHAR_FREQ_NUM){
                    unsigned int old_freq_num = CHAR_FREQ_NUM;
                    CHAR_FREQ_NUM *=2;
                    printf("Resize\n");
                    pCharFreq = realloc(pCharFreq, CHAR_FREQ_NUM * sizeof(int));
                    //Initialize to 0
                    for(unsigned int i = old_freq_num; i < CHAR_FREQ_NUM; i++){
                        pCharFreq[i] = 0;
                    }
                }
                pCharFreq[nCharacters-1]+=1;
                pConsonantFreq[nCharacters-1][nConsonants]+=1;
                //update largestWord???
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
            if(nCharacters > CHAR_FREQ_NUM){
                unsigned int old_freq_num = CHAR_FREQ_NUM;
                CHAR_FREQ_NUM *=2;
                printf("Resize\n");
                pCharFreq = realloc(pCharFreq, CHAR_FREQ_NUM * sizeof(int));
                //Initialize to 0
                for(unsigned int i = old_freq_num; i < CHAR_FREQ_NUM; i++){
                    pCharFreq[i] = 0;
                }
            }
            pCharFreq[nCharacters-1]+=1;
            pConsonantFreq[nCharacters-1][nConsonants]+=1;
            //update largestWord???
            nCharacters = 0;
            nWords+=1;
        }

        printf("File %s\n",argv[textIdx]);
        printf("Number of words:%d\n",nWords);
        printf("Number of consonants:%d\n",nConsonants);
        printf("Word Length:\n");
        //Lines
        printf("%-3s", "");
        for(unsigned int i = 0; i < CHAR_FREQ_NUM; i++){
            printf("%-5d",i+1);
        }
        printf("\n");
        //CharFreq
        printf("%-3s", "");
        for(unsigned int i = 0; i < CHAR_FREQ_NUM; i++){
            printf("%-5d",pCharFreq[i]);
        }
        printf("\n");
        //CharFreq(%)
        printf("%-3s", "");
        for(unsigned int i = 0; i < CHAR_FREQ_NUM; i++){
            printf("%-5.1f",((double)pCharFreq[i] / (double)nWords) * 100);
        }
        printf("\n");

        //ConsonantFreq matrix
        for(unsigned int i1 = 0; i1 < CHAR_FREQ_NUM; i1++){
            printf("%-3d", i1);
            for(int i2 = 0; i2 < 10; i2++){
                printf("%-5d",pConsonantFreq[i1][i2]);
            }
            printf("\n");
        }

        fclose(pFile);
        free(pCharFreq);
        //TODO:Consonants freqs seem wrong
    }
}