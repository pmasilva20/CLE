#include <stdbool.h>
#include <stdlib.h>
#include <stdio.h>
#include "structures.h"

/**
 * Print in the terminal the results stored 
 * \param filesToProcess Number of Files
 */
void PrintResults(struct FileMatrices filesToProcess){
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
