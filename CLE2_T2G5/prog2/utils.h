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


/**
 * Print in the terminal the results stored 
 * \param filesToProcess Number of Files
 */
extern void printResults(struct FileMatrices filesToProcess);


/**
 * Calculate Matrix Determinant
 * \param size Order of the Matrix
 * \param matrix Matrix
 * \return Determinant of the Matrix after Gaussian Elimination
 */
extern double calculateMatrixDeterminant(int orderMatrix,double matrix[orderMatrix][orderMatrix]);