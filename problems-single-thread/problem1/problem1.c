#include "./problem1_functions.h"
#include <stdio.h>
#include <stdlib.h>
#include <wchar.h>
#include <locale.h>
#include <stdbool.h>
#include <time.h>


int main (int argc, char** argv){

    double time0, time1, timeTotal;

    timeTotal = 0.0;

    for(int textIdx = 1; textIdx < argc; textIdx++){
        //Vars needed
        int nWords = 0;
        int nVowelStartWords = 0;
        int nConsonantEndWord = 0;

        time0 = (double) clock() / CLOCKS_PER_SEC;
        int error_code = problem1(argv[textIdx],&nWords,&nVowelStartWords,&nConsonantEndWord);
        time1 = (double) clock() / CLOCKS_PER_SEC;
        timeTotal += (time1 - time0);
        if(error_code != 0){
            printf("Error during file processing of %s",argv[textIdx]);
            continue;
        }

        printf("File name: %s\n",argv[textIdx]);
        printf("Total number of words = %d\n",nWords);
        printf("N. of words beginning with a vowel = %d\n",nVowelStartWords);
        printf("N. of words ending with a consonant = %d\n",nConsonantEndWord);
        printf("\n");
    }
    printf ("Elapsed time = %.6f s\n", timeTotal);
}