/**
 *  \file utils.c 
 *
 *  \brief Assignment 2 : Problem 1 - Counting WOrds
 *
 *  Methods/Operations used by Dispatcher/Workers
 *  
 *  Dispatcher Methods:
 *      \li printResults
 * 
 *
 *  \author João Soares (93078) & Pedro Silva (93011)
*/


#include <stdbool.h>
#include <stdlib.h>
#include <stdio.h>
#include "structures.h"

/**
 * Print in the terminal the results stored 
 * \param filesToProcess File Results Structure
 */
void printResults(struct FileText results){

    /** Get all Files statistics and print to console */
    printf("\n");
    printf("File name: %s\n",results.name);
    printf("Total number of words = %d\n",results.nWords);
    printf("N. of words beginning with a vowel = %d\n",results.nVowelStartWords);
    printf("N. of words ending with a consonant = %d\n",results.nConsonantEndWord);

    printf("\n");

}