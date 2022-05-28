/**
 *  \file utils.h 
 *
 *  \brief Assignment 2 : Problem 2 - Determinant of a Square Matrix
 *
 *  Methods/Operations used by Dispatcher/Workers
 *  
 *  Dispatcher Methods:
 *      \li printResults
 * 
 *  Worker Methods:
 *      \li calculateMatrixDeterminant
 *
 *  \author João Soares (93078) & Pedro Silva (93011)
*/


#include "structures.h"
#include <stdbool.h>

/**
 * Print in the terminal the results stored 
 * \param filesToProcess Number of Files
 */
extern void printResults(struct FileText results);
