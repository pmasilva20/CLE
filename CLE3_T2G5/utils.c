/**
 *  \file utils.c 
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


#include <stdbool.h>
#include <stdlib.h>
#include <stdio.h>
#include "structures.h"

/**
 * Print in the terminal the results stored 
 * \param filesToProcess Number of Files
 */
void printResults(struct FileMatrices filesToProcess){
    printf("\nFile: %s\n",filesToProcess.name);
    if(filesToProcess.numberOfMatrices) {
        for (int a = 0; a < filesToProcess.numberOfMatrices; a++) {
            printf("Matrix %d :\n", filesToProcess.determinant_result[a].id + 1);
            printf("The determinant is %.3e\n", filesToProcess.determinant_result[a].determinant);
        }

        /** Free allocated memory */
        free(filesToProcess.determinant_result);
    }
    else{
        printf("Error Reading File\n");
    }
    printf("\n");

}

/**
 * Calculate Matrix Determinant
 * \param size Order of the Matrix
 * \param matrix Matrix
 * \return Determinant of the Matrix after Gaussian Elimination
 */
double calculateMatrixDeterminant(int orderMatrix,double matrix[orderMatrix][orderMatrix]){

    /** \brief Apply Gaussian Elimination
     *  Generic square matrix of order n into an equivalent upper triangular matrix
    */
    for(int i=0;i<orderMatrix-1;i++){
        //Begin Gauss Elimination
        for(int k=i+1;k<orderMatrix;k++){
            double pivot = matrix[i][i];
            double term=matrix[k][i]/pivot;
            for(int j=0;j<orderMatrix;j++){
                matrix[k][j]=matrix[k][j]-term*matrix[i][j];
            }
        }
    }

    double determinant=1;

    for (int x = 0; x < orderMatrix; x++){
        determinant*= matrix[x][x];
    }

    return determinant;

}