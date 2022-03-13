#include "./problem1_functions.h"
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>


int assert(int val1,int val2){
    if(val1 == val2){
        return true;
    }
    else{
        return false;
    }

}

int main(){
    //TEST 1
    char* filename1 = "test0.txt";

    int q1WordNum = 13;
    int q1VowelStartNum = 5;
    int q1ConsonantEndNum = 12;

    //Execute prob1 main
    //Verify expected versus gotten values
    int nWords = 0;
    int nVowelStartWords = 0;
    int nConsonantEndWord = 0;

    int error_code = problem1(filename1,&nWords,&nVowelStartWords,&nConsonantEndWord);
    if(error_code != 0){
        printf("Error reading");
        exit(-1);
    }
    int right_asserts = 0;
    right_asserts+= assert(nWords,q1WordNum);
    right_asserts+= assert(nVowelStartWords,q1VowelStartNum);
    right_asserts+= assert(nConsonantEndWord,q1ConsonantEndNum);
    printf("Got %d/3 criteria right\n",right_asserts);
}
