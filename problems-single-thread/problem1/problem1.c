#include "./problem1_functions.h"
#include <stdio.h>
#include <stdlib.h>
#include <wchar.h>
#include <locale.h>
#include <stdbool.h>


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